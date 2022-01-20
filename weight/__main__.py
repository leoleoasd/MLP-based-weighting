import random
import os

import pytorch_lightning as pl
import torch
from argparse import ArgumentParser

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from IPython import embed

from .dataset import WeightDataModule
from .model import WeightingModel


def main():
    try:
        seed = os.environ['PL_GLOBAL_SEED']
        seed = int(seed)
    except Exception:
        seed = random.randint(0, 2147483647)
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--not_train', dest='train', action='store_false')
    parser.add_argument('--not_test', dest='test', action='store_false')
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--test_ckpt_path', type=str, default=None)
    parser.set_defaults(train=True)
    parser.set_defaults(test=True)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = WeightingModel.add_model_specific_args(parser)
    parser = WeightDataModule.add_dataset_specific_args(parser)
    args = parser.parse_args()
    for k in vars(args):
        print(f"{k.ljust(30)} = {args.__getattribute__(k)}")
    pl.seed_everything(args.seed)
    data_module = WeightDataModule(args)
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=False),
        callbacks=[
            # EarlyStopping(monitor="acc/validation", min_delta=0.00, patience=3, verbose=False, mode="max"),
            ModelCheckpoint(
                monitor="acc/validation",
                filename="biomrc-e{epoch:02d}-{step}-acc{acc/validation:.4f}",
                save_top_k=3,
                mode="max",
                save_on_train_epoch_end=True,
                auto_insert_metric_name=False,
            )
        ],
        logger=TensorBoardLogger("weighting_logs", name=args.name, version=args.version,
                                 default_hp_metric=False) if args.train else None,
    )
    if args.train:
        model = WeightingModel(args.learning_rate)
        trainer.fit(model, data_module)
    else:
        model = WeightingModel.load_from_checkpoint(args.test_ckpt_path)
    if args.test:
        result = trainer.test(model=model, datamodule=data_module)
        print(result)

if __name__ == '__main__':
    main()
