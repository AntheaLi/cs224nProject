import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForQuestionAnswering
from util import kl_coef


class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=6, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, hidden_size=768, intermediate_size=3072):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.LeakyReLU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LeakyReLU(),
            nn.Linear(intermediate_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward the discriminator."""
        out = self.layer(x)
        return out

def MMD(source, target):
    mmd_loss = torch.exp(-1 / (source.mean(dim=0) - target.mean(dim=0)).norm())
    return mmd_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertConfig
from utils import kl_coef


class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=6, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob


class DomainQA(nn.Module):
    def __init__(self, num_classes=6, hidden_size=768,
                 num_layers=3, dropout=0.1, dis_lambda=0.5, concat=False, anneal=False):
        super(DomainQA, self).__init__()

        self.distilbert = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
        self.config = self.distilbert.config
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        self.config.output_scores = True
        self.WEIGHTS_NAME="DomainQA"

        self.qa_outputs = nn.Linear(hidden_size, 2)
        # init weight
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()
        if concat:
            input_size = 2 * hidden_size
        else:
            input_size = hidden_size
        self.discriminator = DomainDiscriminator(num_classes, input_size, hidden_size, num_layers, dropout)

        self.num_classes = num_classes
        self.dis_lambda = dis_lambda
        self.anneal = anneal
        self.concat = concat
        self.sep_id = 102

    # only for prediction
    def forward(self, input_ids, attention_mask,
                start_positions=None, end_positions=None, labels=None,
                dtype=None, global_step=22000):
        if dtype == "qa":
            qa_loss = self.forward_qa(input_ids, attention_mask,
                                      start_positions, end_positions, global_step)
            return qa_loss

        elif dtype == "dis":
            assert labels is not None
            dis_loss = self.forward_discriminator(input_ids, attention_mask, labels)
            return dis_loss

        else:
            sequence_output = self.distilbert(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)#, output_all_encoded_layers=False)
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            return start_logits, end_logits

    def forward_qa(self, input_ids, token_type_ids, attention_mask, start_positions, end_positions, global_step):
        sequence_output = self.distilbert(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                          end_positions=end_positions)  # , output_all_encoded_layers=False)
        hidden = sequence_output.hidden[-1]
        hidden = hidden[:, 0]
        log_prob = self.discriminator(hidden)
        targets = torch.ones_like(log_prob) * (1 / self.num_classes)
        # As with NLLLoss, the input given is expected to contain log-probabilities
        # and is not restricted to a 2D Tensor. The targets are given as probabilities
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        if self.anneal:
            self.dis_lambda = self.dis_lambda * kl_coef(global_step)
        kld = self.dis_lambda * kl_criterion(log_prob, targets)

        qa_loss = sequence_output.loss
        total_loss = qa_loss + kld
        return total_loss

    def forward_discriminator(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            sequence_output, _ = self.distilbert(input_ids, attention_mask)#, output_all_encoded_layers=False)
            hidden = sequence_output.hidden[-1]
            hidden = hidden[:, 0]  # [b, d] : [CLS] representation
        log_prob = self.discriminator(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)
        return loss



    def save_pretrained(self, save_directory):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        state_dict = model_to_save.state_dict()

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, self.WEIGHTS_NAME)

        model_to_save.config.save_pretrained(save_directory)
        torch.save(state_dict, output_model_file)


class DistilBertEncoder(nn.Module):
    def __init__(self, hidden_size=768):
        super(DistilBertEncoder, self).__init__()
        self.distilbert = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
        self.config = self.distilbert.config
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        self.config.output_scores = True
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None, start_positions=None, end_positions=None):
        outputs = self.distilbert(x, attention_mask=mask, start_positions=start_positions, end_positions=end_positions)
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, 0]
        feat = self.pooler(pooled_output)
        return feat, outputs

class Network_bak(nn.Module):
    def __init__(self, args, hidden_size=768):
        super(DistilBertEncoder, self).__init__()
        self.args = args
        self.src_G = DistilBertEncoder()
        self.tgt_G = DistilBertEncoder()
        self.config = self.tgt_G.config
        self.D = Discriminator()
        self.ce_criterion = nn.BCELoss()
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, source, target, dtype=None):
        if dtype=='G':
            loss = self.forwar_G(source)
        elif dtype=='D':
            loss = self.forward_D(target)
        else:
            # for evaluation:
            tgt_input, tgt_attention_mask, _, _ = target
            feat, output = self.tgt_G(tgt_input, mask=tgt_attention_mask)
            start_logits = output.start_logits
            end_logits=output.end_logits
            return start_logits, end_logits


    def forward_G(self, source, target):
        self.src_G.eval()
        self.tgt_G.train()
        self.D.train()
        src_x, src_mask, src_start_positions, src_end_positions = source
        tgt_x, tgt_mask, tgt_start_positions, tgt_end_positions = target
        batch_size=src_x.shape[0]

        with torch.no_grad():
            feat_src, output_src = self.src_G(src_x, src_mask, src_start_positions, src_end_positions)

        feat_src_tgt, output_src_tgt = self.tgt_G(src_x, src_mask, src_start_positions, src_end_positions)
        feat_tgt, output_tgt = self.tgt_G(tgt_x, tgt_mask, tgt_start_positions, tgt_end_positions)
        feat_concat = torch.cat([feat_src_tgt, feat_tgt], 0)
        pred_feat = self.D(feat_concat.clone().detach())
        label_src = torch.zeros(batch_size).unsqueeze(1).to(self.args.device)
        label_tgt = torch.ones(batch_size).unsqueeze(1).to(self.args.device)
        label_feat = torch.cat([label_src, label_tgt], 0)
        kl_loss = self.kl_criterion()
        task_loss = output_tgt.loss
        loss = kl_loss + task_loss

        return loss


    def forwar_D(self, source, target):
        self.src_G.eval()
        self.tgt_G.train()
        self.D.train()
        src_x, src_mask, src_start_positions, src_end_positions = source
        tgt_x, tgt_mask, tgt_start_positions, tgt_end_positions = target
        batch_size=src_x.shape[0]

        feat_src_tgt, output_src_tgt = self.tgt_G(src_x, src_mask, src_start_positions, src_end_positions)
        feat_tgt, output_tgt = self.tgt_G(tgt_x, tgt_mask, tgt_start_positions, tgt_end_positions)
        feat_concat = torch.cat([feat_src_tgt, feat_tgt], 0)
        pred_feat = self.D(feat_concat.clone().detach())
        label_src = torch.zeros(batch_size).unsqueeze(1).to(self.args.device)
        label_tgt = torch.ones(batch_size).unsqueeze(1).to(self.args.device)
        label_feat = torch.cat([label_src, label_tgt], 0)
        dis_loss = self.ce_criterion(pred_feat, label_feat)
        loss = dis_loss
        return loss


    def save_pretrained(self):
        torch.save()




