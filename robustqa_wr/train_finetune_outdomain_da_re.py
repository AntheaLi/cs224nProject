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


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
'''###'''
from args_finetune_outdomain_da_re import get_train_test_args

from tqdm import tqdm

'''###'''
from prior_wd_optim import PriorWD
from transformers import get_linear_schedule_with_warmup

#### Change Made By Xuran Wang: Add lines #######
import xuran_perform_eda

train_fraction = 1

#### Change End #######

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
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
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
    def __init__(self, args, log):
        self.lr = args.lr

        '''###'''
        self.adam_epsilon = args.adam_epsilon
        self.prior_weight_decay = args.prior_weight_decay
        self.max_steps = args.max_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.warmup_ratio = args.warmup_ratio
        self.warmup_steps = args.warmup_steps
        self.layerwise_learning_rate_decay = args.layerwise_learning_rate_decay
        self.weight_decay = args.weight_decay

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

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
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
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

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

    '''###'''
    # http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
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
        '''
        A list of dictionaries (named 'group') which gives a simple way of breaking a model’s parameters into separate components for optimization.
        One use for multiple param_groups would be in training separate layers of a network using, for example, different learning rates. 
        ###Another prominent use cases arises in transfer learning. When fine-tuning a pretrained network, you may want to gradually 
        unfreeze layers and add them to the optimization process as finetuning progresses. For this, param_groups are vital.###
        '''

    def train(self, model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)

        # optim = AdamW(model.parameters(), lr=self.lr)

        '''###'''
        # Prepare optimizer and schedule (linear warmup and decay)
        if self.max_steps > 0:
            t_total = self.max_steps  # Total number of training updates
            self.num_epochs = self.max_steps // (len(train_dataloader) // self.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.gradient_accumulation_steps * self.num_epochs   # self.gradient_accumulation_steps = 1

        if self.warmup_ratio > 0:
            assert self.warmup_steps == 0
            self.warmup_steps = int(self.warmup_ratio * t_total)

        # model.parameters() or optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model)
        ''' 
        model.parameters() => This iterable must have a deterministic ordering - the user of your optimizer shouldn’t 
        pass in something like a dictionary or a set. Usually a list of torch.Tensor objects is given.
        
        params => self.param_groups: A list of dictionaries (named 'group') which gives a simple way of breaking a 
        model’s parameters into separate components for optimization. A prominent use cases arises in transfer 
        learning. When fine-tuning a pretrained network, you may want to gradually unfreeze layers and add them to 
        the optimization process as fine-tuning progresses.
        '''
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(model)
        optim = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.adam_epsilon)  # correct_bias: bool = True
        optimizer = PriorWD(optim, use_prior_wd=self.prior_weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
        )

        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    # optim.zero_grad()

                    '''###'''
                    optimizer.zero_grad()  # Same as model.zero_grad()

                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    # optim.step()

                    '''###'''
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule

                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
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
                            self.save(model)
                    global_idx += 1
        return best_scores

def get_dataset_eda_revised(args, datasets, data_dir, tokenizer, split_name, train_fraction):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        # dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict_curr = xuran_perform_eda.perform_eda(f'{data_dir}/{dataset}', dataset, train_fraction)
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)

    #### Change Made By Xuran Wang: Comment out original lines #######

    # model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    #### Change End #######



    #### Change Made By Xuran Wang: Add custom lines #######

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    finetuned_model_path = 'save/baseline-01/'

    #### Change End #######


    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")


        #### Change Made By Xuran Wang: Add custom lines #######

        checkpoint_path = os.path.join(finetuned_model_path, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)

        #### Change End #######

        '''###'''
        # if args.reinit_pooler:
        #     encoder_temp = getattr(model, "distilbert")  # Equivalent to model.distilbert
        #     encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
        #     encoder_temp.pooler.dense.bias.data.zero_()  # The change of encoder_temp would affect the model
        #     for p in encoder_temp.pooler.parameters():
        #         p.requires_grad = True


        if args.reinit_layers > 0:
            import torch.nn as nn
            from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention, FFN
            # model_distilbert = getattr(model, "distilbert")  # model.distilbert; change of model_distilbert affects model!
            # Reinitialization for the last few layers
            for layer in model.distilbert.transformer.layer[-args.reinit_layers:]:
                for module in layer.modules():
                    # print(module)
                    model.distilbert._init_weights(module)  # It's the line equivalent to below approach
                    # if isinstance(module, nn.modules.linear.Linear):  # Original form for nn.Linear
                    #     # model.config.initializer_range == model.distilbert.config.initializer_range => True
                    #     module.weight.data.normal_(mean=0.0, std=model.distilbert.config.initializer_range)
                    #     if module.bias is not None:
                    #         module.bias.data.zero_()
                    # elif isinstance(module, nn.modules.normalization.LayerNorm):
                    #     module.weight.data.fill_(1.0)
                    #     module.bias.data.zero_()
                    # elif isinstance(module, FFN):
                    #     for param in [module.lin1, module.lin2]:
                    #         param.weight.data.normal_(mean=0.0, std=model.distilbert.config.initializer_range)
                    #         if param.bias is not None:
                    #             param.bias.data.zero_()
                    # elif isinstance(module, MultiHeadSelfAttention):
                    #     for param in [module.q_lin, module.k_lin, module.v_lin, module.out_lin]:
                    #         param.data.weight.normal_(mean=0.0, std=model.distilbert.config.initializer_range)
                    #         if param.bias is not None:
                    #             param.bias.data.zero_()


        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(args.device)

        trainer = Trainer(args, log)

        #### Change Made By Xuran Wang: Add custom lines, comment out original line #######

        # train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')

        train_dataset, _ = get_dataset_eda_revised(args, args.train_datasets, args.train_dir, tokenizer, 'train', train_fraction)

         #### Change End #######

        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
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
