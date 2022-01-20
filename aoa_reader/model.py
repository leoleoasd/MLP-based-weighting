from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.profiler import PassThroughProfiler
# from pytorch_pretrained_bert import BertModel
from transformers import BertModel, BertTokenizer
from IPython import embed


class AOAReader_Model(LightningModule):
    acc_list = []
    acc_epoch_list = []

    def training_epoch_end(self, outputs):
        self.log("acc/train_epoch_avg", torch.mean(torch.tensor(self.acc_epoch_list)))
        self.acc_epoch_list = []

    def __init__(self, learning_rate, embedding_dim, hidden_dim, dropout_prob=0.2, random_seed=None,
                 bert_learning_rate=1e-5, bert_dir = "", flood = 0.0, occ = 'sum', tok = 'sum', profiler=None):
        super(AOAReader_Model, self).__init__()
        self.save_hyperparameters()
        self.profiler = profiler or PassThroughProfiler()
        self.bert = BertModel.from_pretrained(self.hparams.bert_dir)
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.bert_dir)
        # self.word_embeddings = nn.Embedding(self.hparams.vocab_size, self.hparams.embedding_dim)
        self.context_h = torch.nn.Parameter(torch.randn(2, 1, self.hparams.hidden_dim))
        torch.nn.init.xavier_normal_(self.context_h)
        self.context_bigru = nn.GRU(
            input_size=self.hparams.embedding_dim,
            hidden_size=self.hparams.hidden_dim,
            num_layers=1,
            bidirectional=True,
            bias=True,
            dropout=0,
            batch_first=True
        )
        self.question_h = torch.nn.Parameter(torch.randn(2, 1, self.hparams.hidden_dim))
        torch.nn.init.xavier_normal_(self.question_h)
        self.question_bigru = nn.GRU(
            input_size=self.hparams.embedding_dim,
            hidden_size=self.hparams.hidden_dim,
            num_layers=1,
            bidirectional=True,
            bias=True,
            dropout=0,
            batch_first=True
        )
        self.softmax = torch.nn.Softmax()
        self.dropout_f = nn.Dropout(p=self.hparams.dropout_prob)

    def get_candidates_score(self, pws, context, candidates):
        with self.profiler.profile("get_candidates_score"):

            # for each candidate, for each candidate embedding, for each position, we compare context with candidate.
            masks = torch.eq(context.unsqueeze(1).unsqueeze(1), candidates.unsqueeze(3))
            # masks shape: [batch_size, max_candidate_count, max_candidate_embedding_length, sequence_length]
            # we apply mask on pws, which gave us a score of each candidate, each embedding, each position's score.
            # pws shape: [batch_size, sequence_length]
            # mul shape example: [20, 14, 33, 512]
            score = torch.mul(masks, pws.unsqueeze(1).unsqueeze(1))
            # score shape: [batch_size, max_candidate_count, max_candidate_embedding_length, sequence_length]
            # sum 3: all occurrence
            # sum 2: all embedding: can replace with max.
            # score = score.sum(3).sum(2)
            if self.hparams.occ == 'sum':
                score = score.sum(3)
            elif self.hparams.occ == 'max':
                score = score.max(3)[0]
            else:
                raise Exception("illegal occ")
            if self.hparams.tok == 'sum':
                score = score.sum(2)
            elif self.hparams.tok == 'max':
                score = score.max(2)[0]
            else:
                raise Exception("illegal tok")
            
            # score shape: [batch_size, max_candidate_count]
            return score

    def calculate_accuracy(self, soft_res, target):
        with self.profiler.profile("calculate_accuracy"):
            total = (soft_res.size(0) * 1.0)
            soft_res = torch.argmax(soft_res.data, dim=1)
            target = target
            wright_ones = torch.sum(soft_res == target)
            acc = wright_ones / total
            return acc

    def get_pairwise_score(self, con_gru_out, quest_gru_out):
        with self.profiler.profile("get_pairwise_score"):
            rows_att = []
            cols_att = []
            M = torch.bmm(con_gru_out, quest_gru_out.transpose(1, 2)) # mat-mul
            rows_att = torch.softmax(M, 2) # row-wise softmax
            cols_att = torch.softmax(M, 1) # col-wise softmax
            av = rows_att.sum(1) / (rows_att.size(1) * 1.0)
            return torch.bmm(cols_att, av.unsqueeze(2)).squeeze(-1)

    def forward(self, input_ids, attention_mask, token_type_ids, context, question, candidates, target):
        embeds = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        attention_mask = attention_mask[:, :, None]
        token_type_ids = token_type_ids[:, :, None]
        cont_embeds = embeds.masked_fill(
            torch.logical_and(
                attention_mask, token_type_ids), 0)
        quest_embeds = embeds.masked_fill(
            torch.logical_and(
                attention_mask, 1 - token_type_ids), 0)
        cont_embeds = self.dropout_f(cont_embeds)
        quest_embeds = self.dropout_f(quest_embeds)

        context_out, context_hn = self.context_bigru(cont_embeds)
        question_out, question_hn = self.question_bigru(quest_embeds)
        pws = self.get_pairwise_score(context_out, question_out)
        pws_cands = self.get_candidates_score(pws, context, candidates)
        log_soft_res = F.log_softmax(pws_cands)
        soft_res = F.softmax(pws_cands)
        acc = self.calculate_accuracy(log_soft_res, target)
        losses = F.nll_loss(log_soft_res, target, weight=None, size_average=True)
        losses = (losses - self.hparams.flood).abs() + self.hparams.flood # flood
        return losses, acc, log_soft_res, soft_res

    def configure_optimizers(self):
        # Another learning rate for embedding layer.
        bert_params = list(self.bert.parameters())  # list(filter(lambda kv: kv[0] in my_list, self.named_parameters()))
        base_params = [self.context_h.data, self.question_h.data, *self.context_bigru.parameters(),
                       *self.question_bigru.parameters()]
        optimizer = torch.optim.Adam([
            {"params": base_params},
            {"params": bert_params, "lr": self.hparams.bert_learning_rate},
        ], lr=self.hparams.learning_rate)
        return optimizer

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams, {"acc/validation": 0, "acc/train_epoch_avg": 0, "acc/test": 0})

    def training_step(self, batch, batch_idx):
        # b_context, b_quest, b_candidates, b_target = batch
        loss, acc_, log_soft_res, soft_res = self(*batch)
        # average acc on last 50 batch
        self.acc_list.append(acc_)
        self.acc_epoch_list.append(acc_)
        if len(self.acc_list) > 50:
            self.acc_list = self.acc_list[-50:]
        self.log("acc/train_avg", torch.mean(torch.tensor(self.acc_list)), prog_bar=True, logger=True)
        self.log("acc/train", acc_, prog_bar=True, logger=True)
        self.log("loss/train", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # b_context, b_quest, b_candidates, b_target = batch
        loss, acc_, log_soft_res, soft_res = self(*batch)
        self.log("acc/validation", acc_, prog_bar=True, logger=True)
        self.log("loss/validation", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # b_context, b_quest, b_candidates, b_target = batch
        loss, acc_, log_soft_res, soft_res = self(*batch)
        self.log("acc/test", acc_, prog_bar=True, logger=True)
        self.log("loss/test", loss, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        # b_context, b_quest, b_candidates, b_target = batch
        loss, acc_, log_soft_res, soft_res = self(*batch)
        # self.log("acc/test", acc_, prog_bar=True, logger=True)
        # self.log("loss/test", loss, prog_bar=True, logger=True)
        return batch[6], soft_res

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--embedding_dim', type=int, default=768)
        parser.add_argument('--dropout_prob', type=float, default=0.2)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--bert_learning_rate', type=float, default=1e-5)
        parser.add_argument('--occ_agg', type=str, default='sum')
        parser.add_argument('--tok_agg', type=str, default='sum')
        return parser
