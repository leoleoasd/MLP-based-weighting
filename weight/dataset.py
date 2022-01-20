from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader


class WeightDataset(torch.utils.data.IterableDataset):
    def __init__(self, aoa_filename, bert_filename, batch_size, begin, end):
        super().__init__()
        aoa = torch.load(aoa_filename)
        self.aoa, self.answer = aoa['outputs'], aoa['answers']
        self.bert = torch.load(bert_filename)
        self.batch_size = batch_size
        self.begin = begin
        self.end = end

    def __iter__(self):
        for i in range(self.begin, self.end, self.batch_size):
            yield self.aoa[i:i + self.batch_size], \
                  self.bert[i:i + self.batch_size], \
                  self.answer[i:i + self.batch_size]


class WeightDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.aoa_filename = args.aoa_filename
        self.bert_filename = args.bert_filename
        self.batch_size = args.batch_size

    def train_dataloader(self):
        return DataLoader(
            WeightDataset(self.aoa_filename, self.bert_filename, self.batch_size, 0, 87500),
            batch_size=None,
            batch_sampler=None,
        )

    def val_dataloader(self):
        return DataLoader(
            WeightDataset(self.aoa_filename, self.bert_filename, self.batch_size, 87500, 87500 + 6250),
            batch_size=None,
            batch_sampler=None,
        )

    def test_dataloader(self):
        return DataLoader(
            WeightDataset(self.aoa_filename, self.bert_filename, self.batch_size, 87500 + 6250, 100000),
            batch_size=None,
            batch_sampler=None,
        )

    def predict_dataloader(self):
        return DataLoader(
            WeightDataset(self.aoa_filename, self.bert_filename, self.batch_size, 0, 100000),
            batch_size=None,
            batch_sampler=None,
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--aoa_filename', type=str, default='predictions.pt')
        parser.add_argument('--bert_filename', type=str, default='scibert_predict.pt')
        parser.add_argument('--batch_size', type=int, default=100)
        return parser
