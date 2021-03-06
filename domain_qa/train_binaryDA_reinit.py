import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter
from distilbert_DANN import DomainQA

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args
import xuran_perform_eda

from tqdm import tqdm

from prior_optim_WD import PriorWD
from transformers import get_linear_schedule_with_warmup

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    tokenized_examples["label"] = []

    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["label"].append(dataset_dict["label"][sample_index])
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    tokenized_examples['label'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        tokenized_examples['label'].append(dataset_dict['label'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples



def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples


#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log, model):
        self.lr = args.lr
        self.args = args

        self.adam_epsilon = args.adam_epsilon
        self.prior_weight_decay = args.prior_weight_decay
        self.max_steps = args.max_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.warmup_ratio = args.warmup_ratio
        self.warmup_steps = args.warmup_steps
        self.layerwise_learning_rate_decay = args.layerwise_learning_rate_decay
        self.weight_decay = args.weight_decay
        self.warmup_steps = args.warmup_steps
        self.t_total = args.t_total

        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self.model = model
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(self.model.distilbert)
        # qa_params = list(self.model.distilbert.parameters())
        dis_params = list(self.model.discriminator.parameters())
        self.qa_optim = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon)
        self.dis_optim = AdamW(dis_params, lr=self.lr)
        self.optimizer = PriorWD(self.qa_optim, use_prior_wd=self.prior_weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total)
    
    
    def get_optimizer_grouped_parameters(self, model):
        no_decay = ["bias", "LayerNorm.weight"]  # Decay on bias is trivial & LayerNorm is direct calculation => reduce training time
        if self.layerwise_learning_rate_decay == 1.0:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    # name, param => "Parameter containing: tensor([...])"
                    "weight_decay": self.weight_decay,
                    "lr": self.lr,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    # No decay for name with "bias" and "LayerNorm.weight"
                    "weight_decay": 0.0,
                    "lr": self.lr,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if "qa_outputs" in n],
                    # No decay for qa_outputs.weight/bias
                    "weight_decay": 0.0,
                    "lr": self.lr,
                },
            ]

            # A list for elements of Embeddings or TransformerBlock
            layers = [getattr(model, "distilbert").embeddings] + list(getattr(model, "distilbert").transformer.layer)
            layers.reverse()
            lr = self.lr
            for layer in layers:
                lr *= self.layerwise_learning_rate_decay
                optimizer_grouped_parameters += [
                    {
                        "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.weight_decay,
                        "lr": lr,
                    },
                    {
                        "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]

        return optimizer_grouped_parameters  # self.param_groups as for Optimizer base class
    def save(self, model):
        model.save_pretrained(self.path)

    def cal_running_avg_loss(self, loss, running_avg_loss, decay=0.99):
        if running_avg_loss == 0:
            return loss
        else:
            running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
            return running_avg_loss

    def evaluate(self, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        self.model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs  = self.model(input_ids, attention_mask)
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # Forward

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        self.model.to(device)
        if self.args.load_weights != '':
            self.model.load_state_dict(torch.load(self.args.load_weights))
            print('loaded pretrained weights ... ')
        self.model.train()

        if self.args.reinit_layers > 0:
            for layer in self.model.distilbert.distilbert.transformer.layer[-self.args.reinit_layers:]:
                for module in layer.modules():
                    self.model.distilbert.distilbert._init_weights(module)  # It's the line equivalent to below approach

        global_idx = 0
        avg_qa_loss = 0
        avg_dis_loss = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)
        step = 1
        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    labels = batch['label'].to(device)
                    ##################################
                    # start adversarial training

                    self.qa_optim.zero_grad()
                    qa_loss = self.model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions, labels=labels, dtype='qa',
                                    global_step=step)
                    qa_loss = qa_loss.mean()
                    qa_loss.backward()
                    avg_g_loss = self.cal_running_avg_loss(qa_loss.item(), avg_qa_loss)
                    self.optimizer.step()
                    self.scheduler.step()

                    if self.args.wasserstein:
                        for p in self.model.discriminator.parameters():
                            p.data.clamp_(self.args.clamp_lower, self.args.clamp_upper)

                    self.dis_optim.zero_grad()
                    dis_loss = self.model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions, labels=labels, dtype='dis',
                                    global_step=step)
                    dis_loss = dis_loss.mean()
                    dis_loss.backward()

                    avg_dis_loss = self.cal_running_avg_loss(dis_loss.item(), avg_dis_loss)
                    self.dis_optim.step()

                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, qa=qa_loss.item(), dis=dis_loss.item())
                    tbx.add_scalar('train/qa', qa_loss.item(), global_idx)
                    tbx.add_scalar('train/dis', dis_loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate( eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(self.model)
                    global_idx += 1
        return best_scores

def get_train_dataset(args, target_data_dir, target_dataset, tokenizer, split_name, source_data_dir=None, source_dataset=None):
    dataset_dict_source = None
    dataset_dict_target = None
    data_encodings_source = None
    source_dataset_name = 'binary'
    target_dataset_name = f'binary{args.data_suffix}'
    if source_data_dir is not None and source_dataset is not None:
        datasets = source_dataset.split(',')  
        for dataset in datasets:
            source_dataset_name += f'_{dataset}'
            dataset_dict_curr = util.read_squad(f'{source_data_dir}/{dataset}',label=0)
            dataset_dict_source = util.merge(dataset_dict_source, dataset_dict_curr)
        data_encodings_source = read_and_process(args, tokenizer, dataset_dict_source, source_data_dir, source_dataset_name, split_name)
    datasets = target_dataset.split(',')
    for dataset in datasets:
        target_dataset_name += f'_{dataset}'
        # dataset_dict_curr = util.read_squad(f'{target_data_dir}/{dataset}', label=1)
        dataset_dict_curr = xuran_perform_eda.perform_eda(f'{target_data_dir}/{dataset}', dataset, train_fraction=1, label=1)
        dataset_dict_target = util.merge(dataset_dict_target, dataset_dict_curr)
    data_encodings_target = read_and_process(args, tokenizer, dataset_dict_target, target_data_dir, target_dataset_name, split_name)
    dataset_dict = util.merge(dataset_dict_source, dataset_dict_target)
    data_encodings = util.merge(data_encodings_source, data_encodings_target)
    return util.QADomainDataset(data_encodings, train=(split_name=='train')), dataset_dict

def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name='binary'
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}', label=1)
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADomainDataset(data_encodings, train=(split_name=='train')), dataset_dict

def main():
    # define parser and arguments
    args = get_train_test_args()
    util.set_seed(args.seed)
    # model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model =  DomainQA(args.num_classes,
                      args.hidden_size, args.num_layers,
                      args.dropout, args.dis_lambda,
                      args.concat, args.anneal)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if args.load_weights != '':
            args.load_weights = os.path.join(args.load_weights, 'checkpoint', model.WEIGHTS_NAME)
            model.load_state_dict(torch.load(args.load_weights))
        if args.load_distilbert_weights != '':
            args.load_distilbert_weights = os.path.join(args.load_distilbert_weights, 'checkpoint', model.WEIGHTS_NAME)
            model.distilbert.load_state_dict(torch.load(args.load_distilbert_weights))
            print('loaded pretrained distilbert weights from', args.load_distilbert_weights)

        #target_data_dir, target_dataset, tokenizer, split_name, source_data_dir = None, source_dataset = None
        train_dataset, _ = get_train_dataset(args, \
                                       args.target_train_dir,\
                                       args.target_train_datasets,\
                                       tokenizer, 'train', \
                                       source_data_dir=args.source_train_dir, \
                                       source_dataset=args.source_train_datasets)
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, \
                                       args.eval_datasets,\
                                       args.eval_dir,\
                                       tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
                # warm up
        if args.max_steps > 0:
            args.t_total = args.max_steps  # Total number of training updates
            args.num_epochs = args.max_steps // (len(train_loader) // args.gradient_accumulation_steps) + 1
        else:
            args.t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_epochs   # self.gradient_accumulation_steps = 1

        if args.warmup_ratio > 0:
            assert args.warmup_steps == 0
            args.warmup_steps = int(args.warmup_ratio * args.t_total)

        trainer = Trainer(args, log, model)

        best_scores = trainer.train(train_loader, val_loader, val_dict)


    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log, model)
        config_path = os.path.join(args.save_dir, 'checkpoint', 'config.json')
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint', model.WEIGHTS_NAME)
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
