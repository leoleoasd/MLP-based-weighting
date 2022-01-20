#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

__author__ = 'Petros'

my_seed = 1989
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import os
import sys
import numpy as np
import pickle
import torch.backends.cudnn as cudnn
import random
import json
import copy
from pytorch_pretrained_bert import BertTokenizer, BertModel
from tqdm import tqdm
from nltk import sent_tokenize
from IPython import embed
from argparse import ArgumentParser
import gc

random.seed(my_seed)
torch.manual_seed(my_seed)
np.random.seed(my_seed)

cudnn.benchmark = True
doTrain = True
doEval = False

embedding_dim = 30
hidden_dim = 100
gpu_device = 0
use_cuda = torch.cuda.is_available()
if (use_cuda):
    torch.cuda.manual_seed(my_seed)
    print("Using GPU")

parser = ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default="./scibert_best.pt")
args = parser.parse_args()


def print_params(show_model=True):
    if show_model:
        print(40 * '=')
        print(model)
    print(40 * '=')
    total_params = 0
    print('Trainable Parameters\n')
    for parameter in model.parameters():
        if parameter.requires_grad:
            v = 1
            for s in parameter.size():
                v *= s
            total_params += v
    print(40 * '=')
    print(total_params)
    print(40 * '=')


class SciBertReaderMax(nn.Module):
    def __init__(self, frozen_top):
        super(SciBertReaderMax, self).__init__()

        self.tok = BertTokenizer.from_pretrained('data/scibert')
        self.bert = BertModel.from_pretrained('data/scibert')
        self.linear = nn.Linear(2 * 768, 100, bias=True)
        self.linear2 = nn.Linear(100, 1, bias=True)

        self.frozen_top = frozen_top

        for p in self.bert.parameters():
            p.requires_grad = False

        if use_cuda:
            self.bert = self.bert.cuda(gpu_device)
            self.linear = self.linear.cuda(gpu_device)
            self.linear2 = self.linear2.cuda(gpu_device)

    def freeze_top(self, optim):
        for p in self.bert.encoder.layer[-1].parameters():
            p.requires_grad = False
        for g in optim.param_groups:
            g['lr'] = 0.001
        self.frozen_top = True

    def unfreeze_top(self, optim):
        for p in self.bert.encoder.layer[-1].parameters():
            p.requires_grad = True
        for g in optim.param_groups:
            g['lr'] = 0.0001
        self.frozen_top = False

    def fix_input(self, abstract, title):
        ab_sents = sent_tokenize(abstract, 'english')
        ab_tok = [self.tok.tokenize(ab) for ab in ab_sents]
        entity_indices = list()
        entity_texts = list()
        for i in range(len(ab_tok)):
            ab_tok[i].insert(0, '[CLS]')
            entity_indices.append(list())
            entity_texts.append(list())
            for j in range(len(ab_tok[i])):
                if ab_tok[i][j] == '@':
                    if ab_tok[i][j + 1] == 'entity':
                        entity_indices[-1].append(j)
            for j in entity_indices[-1]:
                n = j + 1
                while ab_tok[i][n].startswith('##') or ab_tok[i][n] == 'entity':
                    n += 1
                    if n >= len(ab_tok[i]):
                        break
                entity_texts[-1].append(''.join(ab_tok[i][j:n]).replace('##', ''))
        ti_tok = self.tok.tokenize(title.replace('.XXXX', ' [MASK]').replace('XXXX', '[MASK]'))
        ti_tok.insert(0, '[SEP]')
        combined = list()
        mask_indices = list()
        for i in range(len(ab_tok)):
            combined.append(list())
            combined[i].extend(ab_tok[i])
            combined[i].extend(ti_tok)
            mask_indices.append(combined[i].index('[MASK]'))
        combined_inp = [torch.LongTensor(self.tok.convert_tokens_to_ids(c)).unsqueeze(dim=0) for c in combined]

        if use_cuda:
            combined_inp = [e.cuda(gpu_device) for e in combined_inp]

        return combined_inp, mask_indices, entity_indices, entity_texts

    def forward(self, abstract, title, entity_list, answer, ignore_big=True):
        combined_input, mask_indices, entity_indices, entity_texts = self.fix_input(abstract, title)
        max_len = max([e.shape[-1] for e in combined_input])
        if ignore_big and max_len > 512:
            return None, None
        # Pad combined input
        combined_input = torch.stack(
            [F.pad(e, [0, max_len - e.shape[-1], 0, 0], 'constant', 0) for e in combined_input], dim=0).squeeze(dim=1)
        if use_cuda:
            combined_input = combined_input.cuda(gpu_device)
        out = list()
        bert_out = self.bert(combined_input)[0][-1]
        for i in range(combined_input.shape[0]):
            if len(entity_indices[i]) == 0:
                continue
            if len(entity_indices[i]) != 1:
                bert_out_entities = bert_out[i, entity_indices[i], :].squeeze(dim=0)
            else:
                bert_out_entities = bert_out[i, entity_indices[i], :]
            bert_out_mask = bert_out[i, mask_indices[i], :]
            bert_out_mask = bert_out_mask.expand_as(bert_out_entities)
            bert_out_concat = torch.cat([bert_out_entities, bert_out_mask], dim=-1)
            out.append(self.linear2(F.relu(self.linear(bert_out_concat))))
        entity_texts = [e for e in entity_texts if len(e) != 0]
        # Predict also
        preds = dict()
        for et, ot in zip(entity_texts, out):
            ot = ot.detach().cpu().numpy().tolist()
            for e, o in zip(et, ot):
                if e not in preds:
                    preds[e] = float('-inf')
                if o[0] > preds[e]:
                    preds[e] = o[0]
        entity_outs = dict()
        for ent in entity_list:
            entity_outs[ent] = list()
            for r_i, r in enumerate(entity_texts):
                for c_i, c in enumerate(r):
                    if c == ent:
                        entity_outs[ent].append(out[r_i][c_i])
        for ent in list(entity_outs):
            if len(entity_outs[ent]) == 0:
                del entity_outs[ent]
            else:
                entity_outs[ent] = torch.max(torch.cat(entity_outs[ent]))
        return torch.stack(list(entity_outs.values())), entity_outs.keys(), preds

    def predict(self, abstract, title, entity_list, ignore_big=True):
        combined_input, mask_indices, entity_indices, entity_texts = self.fix_input(abstract, title)
        max_len = max([e.shape[-1] for e in combined_input])
        if ignore_big and max_len > 512:
            return None
        # Pad combined input
        combined_input = torch.stack(
            [F.pad(e, [0, max_len - e.shape[-1], 0, 0], 'constant', 0) for e in combined_input], dim=0).squeeze(dim=1)
        if use_cuda:
            combined_input = combined_input.cuda(gpu_device)
        out = list()
        bert_out = self.bert(combined_input)[0][-1]
        for i in range(combined_input.shape[0]):
            if len(entity_indices[i]) == 0:
                continue
            if len(entity_indices[i]) != 1:
                bert_out_entities = bert_out[i, entity_indices[i], :].squeeze(dim=0)
            else:
                bert_out_entities = bert_out[i, entity_indices[i], :]
            bert_out_mask = bert_out[i, mask_indices[i], :]
            bert_out_mask = bert_out_mask.expand_as(bert_out_entities)
            bert_out_concat = torch.cat([bert_out_entities, bert_out_mask], dim=-1)
            out.append(self.linear2(F.relu(self.linear(bert_out_concat))))
        # Find predictions
        entity_texts = [e for e in entity_texts if len(e) != 0]
        preds = dict()
        for et, ot in zip(entity_texts, out):
            ot = ot.detach().cpu().numpy().tolist()
            for e, o in zip(et, ot):
                if e not in preds:
                    preds[e] = float('-inf')
                if o[0] > preds[e]:
                    preds[e] = o[0]
        return preds


# Load checkpoint
resume_from = args.ckpt_path
resumed = False
if os.path.exists(resume_from):
    checkpoint = torch.load(resume_from)
    start_epoch = checkpoint['epoch'] + 1
    best_dev_acc = checkpoint['best_acc']
    best_epoch = checkpoint['best_epoch']
    frozen_t = checkpoint['frozen_top']
    early_stop_counter = checkpoint['early_stop']
    print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))
    print([(e, checkpoint[e]) for e in checkpoint.keys() if e != 'state_dict' and e != 'optimizer'])
    resumed = True
else:
    print('No checkpoint to load!')
    start_epoch = 0
    best_dev_acc = -1
    best_epoch = -1
    early_stop_counter = 0
    frozen_t = True

model = SciBertReaderMax(frozen_t)

if use_cuda:
    model.cuda(gpu_device)

cross_entropy = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print_params()

max_epochs = 40

# Try to resume
if resumed:
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Unfreeze top bert layer if we resumed with unfrozen top
    if not model.frozen_top:
        model.unfreeze_top(optimizer)
        print('BERT Top Layer is Unfrozen')

if doTrain:
    for epoch in range(start_epoch, max_epochs):
        out_f = open('./result.txt', 'a+')
        print()
        out_f.write('\n')
        print('-' * 30)
        out_f.write('-' * 30 + '\n')
        print('\nEpoch {}\n'.format(epoch))
        out_f.write('\nEpoch {}\n\n'.format(epoch))
        print('-' * 30)
        out_f.write('-' * 30 + '\n')
        print()
        out_f.write('\n')
        print_params(show_model=False)
        print()

        model.train()
        for tf in [1, 2, 3, 4, 5, 6, 7]:
            with open('./data/raw/dataset_part{}_small.json'.format(tf)) as fi:
                data = json.load(fi)
            print('Training on file {}'.format(tf))
            out_f.write('Training on file {}'.format(tf) + '\n')
            loss_list = []
            ans_list = []
            pred_list = []
            i = 0
            model.train()
            for ab, ti, en_l, an in tqdm(zip(data['abstracts'], data['titles'], data['entities_list'], data['answers']),
                                         total=len(data['abstracts'])):
                ents = list(map(lambda x: x.split('::')[0].strip(), en_l))
                ans = an.split('::')[0].strip()
                optimizer.zero_grad()
                out, out_ents, preds = model(ab, ti, ents, ans)
                if out is None and out_ents is None:
                    continue

                max_pred = max(preds.keys(), key=(lambda x: preds[x]))
                ans_list.append(ans)
                pred_list.append(max_pred)

                ans_targets = torch.FloatTensor([1.0 if e == ans else 0.0 for e in out_ents])
                if use_cuda:
                    ans_targets = ans_targets.cuda(gpu_device)
                loss = cross_entropy(out.unsqueeze(dim=0), torch.argmax(ans_targets).unsqueeze(dim=0))
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
            train_acc = np.mean(np.asarray(pred_list) == np.asarray(ans_list))
            print('Train Accuracy:', train_acc)
            out_f.write('Train Accuracy: ' + str(train_acc) + '\n')
            print()
            print('Loss:', np.mean(loss_list))
            out_f.write('Loss: ' + str(np.mean(loss_list)) + '\n')
        out_f.close()
        out_f = open('./result.txt', 'a+')
        # Evaluation Dev
        print()
        out_f.write('\n')
        print('Evaluating Dev')
        out_f.write('Evaluating Dev\n')
        with open('./data/raw/dataset_part8_small.json') as fi:
            data = json.load(fi)

        loss_list = []
        ans_list = []
        pred_list = []
        i = 0
        model.eval()
        for ab, ti, en_l, an in tqdm(zip(data['abstracts'], data['titles'], data['entities_list'], data['answers']),
                                     total=len(data['abstracts'])):
            ents = list(map(lambda x: x.split('::')[0].strip(), en_l))
            ans = an.split('::')[0].strip()
            preds = model.predict(ab, ti, ents)
            if preds is None:
                continue
            max_pred = max(preds.keys(), key=(lambda x: preds[x]))
            ans_list.append(ans)
            pred_list.append(max_pred)
        dev_acc = np.mean(np.asarray(pred_list) == np.asarray(ans_list))
        print('Dev Accuracy:', dev_acc)
        out_f.write('Dev Accuracy: ' + str(dev_acc) + '\n')
        do_test = False


        # Save model checkpoint
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_dev_acc,
            'best_epoch': best_epoch,
            'optimizer': optimizer.state_dict(),
            'frozen_top': model.frozen_top,
            'early_stop': early_stop_counter
        }
        torch.save(state,
                   f'./scibert_ckpt_{epoch}_{dev_acc}.pt')

        if dev_acc > best_dev_acc:
            print('Saving best model...')
            out_f.write('Saving best model...' + '\n')
            out_f.close()
            early_stop_counter = 0
            best_dev_acc = dev_acc
            best_epoch = epoch
            do_test = True
            # Save best model
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': best_dev_acc,
                'best_epoch': best_epoch,
                'optimizer': optimizer.state_dict(),
                'frozen_top': model.frozen_top,
                'early_stop': early_stop_counter
            }
            torch.save(state,
                       './scibert_best_ckpt.pt')
        else:
            early_stop_counter += 1
        if early_stop_counter == 5:
            if model.frozen_top:
                print('Unfreezing top BERT Layer')
                out_f.write('Unfreezing top BERT Layer')
                model.unfreeze_top(optimizer)
                print_params(show_model=False)
                early_stop_counter = 0
            else:
                out_f = open('./result.txt',
                             'a+')
                # Early stop
                print('Early stop at epoch', epoch)
                out_f.write('Early stop at epoch ' + str(epoch))
                out_f.close()
                break

        if do_test:
            out_f = open('./result.txt',
                         'a+')
            # Evaluation Test
            print()
            out_f.write('\n')
            print('Evaluating Test')
            out_f.write('Evaluating Test\n')
            with open('./data/raw/dataset_part9_small.json') as fi:
                data = json.load(fi)

            loss_list = []
            ans_list = []
            pred_list = []
            i = 0
            model.eval()
            for ab, ti, en_l, an in tqdm(zip(data['abstracts'], data['titles'], data['entities_list'], data['answers']),
                                         total=len(data['abstracts'])):
                ents = list(map(lambda x: x.split('::')[0].strip(), en_l))
                ans = an.split('::')[0].strip()
                preds = model.predict(ab, ti, ents)
                if preds is None:
                    continue
                max_pred = max(preds.keys(), key=(lambda x: preds[x]))
                ans_list.append(ans)
                pred_list.append(max_pred)
            test_acc = np.mean(np.asarray(pred_list) == np.asarray(ans_list))
            print('Test Accuracy:', test_acc)
            out_f.write('Test Accuracy: ' + str(test_acc) + '\n')
            out_f.close()

if doEval:
    loss_list = []
    ans_list = []
    pred_list = []
    i = 0
    model.eval()
    outs = []
    with tqdm(total=100000) as pbar:
        for tf in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            with open(f'./data/raw/dataset_part{tf}_small.json') as fi:
                data = json.load(fi)
            for ab, ti, en_l, an in zip(data['abstracts'], data['titles'], data['entities_list'], data['answers']):
                ents = list(map(lambda x: x.split('::')[0].strip(), en_l))
                ans = an.split('::')[0].strip()
                out, out_ents, preds = model(ab, ti, ents, an)
                if out is None:
                    print("ERROR")
                    embed()
                    # continue
                outs.append(torch.softmax(out, 0).cpu())
                ans_list.append(ents.index(ans))
                pbar.update(1)
    outs = torch.nn.utils.rnn.pad_sequence(outs, True)
    torch.save(outs, "scibert_predict.pt")
    # test_acc = np.mean(np.asarray(pred_list) == np.asarray(ans_list))
    # print('Test Accuracy:', test_acc)
    # out_f.write('Test Accuracy: ' + str(test_acc) + '\n')
    # out_f.close()
    embed()
