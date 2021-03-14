import json
import random
import os
import logging
import pickle
import string
import re
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict as ddict
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return

def visualize(tbx, pred_dict, gold_dict, step, split, num_visuals):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
        pred_dict (dict): dict of predictions of the form id -> pred.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals <= 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)
    id2index = {curr_id : idx for idx, curr_id in enumerate(gold_dict['id'])}
    visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)
    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_] or 'N/A'
        idx_gold_dict = id2index[id_]
        question = gold_dict['question'][idx_gold_dict]
        context = gold_dict['context'][idx_gold_dict]
        answers = gold_dict['answer'][idx_gold_dict]
        gold = answers['text'][0] if answers else 'N/A'
        tbl_fmt = (f'- **Question:** {question}\n'
                   + f'- **Context:** {context}\n'
                   + f'- **Answer:** {gold}\n'
                   + f'- **Prediction:** {pred}')
        tbx.add_text(tag=f'{split}/{i+1}_of_{num_visuals}',
                     text_string=tbl_fmt,
                     global_step=step)


def get_save_dir(base_dir, name, id_max=100):
    for uid in range(1, id_max):
        save_dir = os.path.join(base_dir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def filter_encodings(encodings):
    filter_idx = [idx for idx, val in enumerate(encodings['end_positions'])
                 if not val]
    filter_idx = set(filter_idx)
    encodings_filtered = {key : [] for key in encodings}
    sz = len(encodings['input_ids'])
    for idx in range(sz):
        if idx not in filter_idx:
            for key in encodings:
                encodings_filtered[key].append(encodings[key][idx])
    return encodings_filtered

def merge(encodings, new_encoding):
    if not encodings:
        return new_encoding
    else:
        for key in new_encoding:
            encodings[key] += new_encoding[key]
        return encodings

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, f'{name}.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

class QADataset(Dataset):
    def __init__(self, encodings, train=True):
        self.encodings = encodings
        self.keys = ['input_ids', 'attention_mask']
        if train:
            self.keys += ['start_positions', 'end_positions']
        assert(all(key in self.encodings for key in self.keys))

    def __getitem__(self, idx):
        return {key : torch.tensor(self.encodings[key][idx]) for key in self.keys}

    def __len__(self):
        return len(self.encodings['input_ids'])

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if len(qa['answers']) == 0:
                    data_dict['question'].append(question)
                    data_dict['context'].append(context)
                    data_dict['id'].append(qa['id'])
                else:
                    for answer in  qa['answers']:
                        data_dict['question'].append(question)
                        data_dict['context'].append(context)
                        data_dict['id'].append(qa['id'])
                        data_dict['answer'].append(answer)
    id_map = ddict(list)
    for idx, qid in enumerate(data_dict['id']):
        id_map[qid].append(idx)

    data_dict_collapsed = {'question': [], 'context': [], 'id': []}
    if data_dict['answer']:
        data_dict_collapsed['answer'] = []
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed['question'].append(data_dict['question'][ex_ids[0]])
        data_dict_collapsed['context'].append(data_dict['context'][ex_ids[0]])
        data_dict_collapsed['id'].append(qid)
        if data_dict['answer']:
            all_answers = [data_dict['answer'][idx] for idx in ex_ids]
            data_dict_collapsed['answer'].append({'answer_start': [answer['answer_start'] for answer in all_answers],
                                                  'text': [answer['text'] for answer in all_answers]})
    return data_dict_collapsed

def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] + 1)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

def convert_tokens(eval_dict, qa_id, y_start_list, y_end_list):
    """Convert predictions to tokens from the context.

    Args:
        eval_dict (dict): Dictionary with eval info for the dataset. This is
            used to perform the mapping from IDs and indices to actual text.
        qa_id (int): List of QA example IDs.
        y_start_list (list): List of start predictions.
        y_end_list (list): List of end predictions.
        no_answer (bool): Questions can have no answer. E.g., SQuAD 2.0.

    Returns:
        pred_dict (dict): Dictionary index IDs -> predicted answer text.
        sub_dict (dict): Dictionary UUIDs -> predicted answer text (submission).
    """
    pred_dict = {}
    sub_dict = {}
    for qid, y_start, y_end in zip(qa_id, y_start_list, y_end_list):
        context = eval_dict[str(qid)]["context"]
        spans = eval_dict[str(qid)]["spans"]
        uuid = eval_dict[str(qid)]["uuid"]
        start_idx = spans[y_start][0]
        end_idx = spans[y_end][1]
        pred_dict[str(qid)] = context[start_idx: end_idx]
        sub_dict[uuid] = context[start_idx: end_idx]
    return pred_dict, sub_dict

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return metric_fn(prediction, '')
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def eval_dicts(gold_dict, pred_dict):
    avna = f1 = em = total = 0
    id2index = {curr_id : idx for idx, curr_id in enumerate(gold_dict['id'])}
    for curr_id in pred_dict:
        total += 1
        index = id2index[curr_id]
        ground_truths = gold_dict['answer'][index]['text']
        prediction = pred_dict[curr_id]
        em += metric_max_over_ground_truths(compute_em, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(compute_f1, prediction, ground_truths)

    eval_dict = {'EM': 100. * em / total,
                 'F1': 100. * f1 / total}
    return eval_dict

def postprocess_qa_predictions(examples, features, predictions,
                               n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = ddict(list)
    for i, feat_id in enumerate(features['id']):
        features_per_example[example_id_to_index[feat_id]].append(i)

    # The dictionaries we have to fill.
    all_predictions = OrderedDict()

    # Let's loop over all the examples!
    for example_index in tqdm(range(len(examples['id']))):
        example = {key : examples[key][example_index] for key in examples}
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            seq_ids = features.sequence_ids(feature_index)
            non_pad_idx = len(seq_ids) - 1
            while not seq_ids[non_pad_idx]:
                non_pad_idx -= 1
            start_logits = start_logits[:non_pad_idx]
            end_logits = end_logits[:non_pad_idx]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features["offset_mapping"][feature_index]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features.get("token_is_max_context", None)
            if token_is_max_context:
                token_is_max_context = token_is_max_context[feature_index]


            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either = 0 or > max_answer_length.
                    if end_index <= start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "start_index": start_index,
                            "end_index": end_index,
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred['offsets']
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0:
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # need to find the best non-empty prediction.
        i = 0
        while i < len(predictions):
            if predictions[i]['text'] != '':
                break
            i += 1
        if i == len(predictions):
            import pdb; pdb.set_trace();

        best_non_null_pred = predictions[i]
        all_predictions[example["id"]] = best_non_null_pred["text"]

    return all_predictions



# All methods below this line are from the official SQuAD 2.0 eval script
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
