import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WeightingModel(pl.LightningModule):

    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(40, 64),
            nn.Tanh(),
            nn.Linear(64, 20)
        )

    def forward(self, aoa, bert, ans):
        out = self.model(torch.cat((aoa, bert), dim=1))
        loss = F.cross_entropy(out, ans)
        acc = torch.sum(torch.argmax(out, 1) == ans) / ans.size(0)
        # loss = (loss - 0.2).abs() + 0.2
        return loss, acc, out

    def training_step(self, batch, batch_idx):
        aoa, bert, ans = batch
        loss, acc, out = self.forward(aoa, bert, ans)
        self.log('loss/train', loss, prog_bar=True, logger=True)
        self.log('acc/train', acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        aoa, bert, ans = batch
        loss, acc, out = self.forward(aoa, bert, ans)
        self.log('loss/validation', loss, prog_bar=True, logger=True)
        self.log('acc/validation', acc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        aoa, bert, ans = batch
        loss, acc, out = self.forward(aoa, bert, ans)
        self.log('loss/test', loss, prog_bar=True, logger=True)
        self.log('acc/test', acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": ReduceLROnPlateau(optimizer),
            #     "monitor": "acc/validation",
            #     "frequency": 1,
            #     # If "monitor" references validation metrics, then "frequency" should be set to a
            #     # multiple of "trainer.check_val_every_n_epoch".
            # },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser
