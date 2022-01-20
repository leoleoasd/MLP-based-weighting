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

from .biomrc_data import BioMRDataModule
from .model import AOAReader_Model


def main():
    try:
        seed = os.environ['PL_GLOBAL_SEED']
        seed = int(seed)
    except Exception:
        seed = random.randint(0, 2147483647)
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--raw_folder', type=str, default="data/raw")
    parser.add_argument('--processed_data_dir', type=str, default="data/processed")
    parser.add_argument('--bert_dir', type=str, default="data/bert_huggingface")
    parser.add_argument('--data_size', type=str, default="small")
    parser.add_argument('--context_threshold', type=int, default=500)
    parser.add_argument('--dataloader_num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--not_train', dest='train', action='store_false')
    parser.add_argument('--test_ckpt_path', type=str, default=None)
    parser.add_argument('--not_test', dest='test', action='store_false')
    parser.add_argument('--predict', dest='predict', action='store_true')
    parser.add_argument('--predict_file_name', type=str, default="predictions.pt")
    parser.add_argument('--flood', type=float, default=0.0)
    parser.add_argument('--pred_data', default=-1, type=int)
    parser.set_defaults(train=True)
    parser.set_defaults(test=True)
    parser.set_defaults(predict=False)

    parser = AOAReader_Model.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1,
                        max_epochs=40,
                        gradient_clip_val=5,
                        flush_logs_every_n_steps=30,
                        log_every_n_steps=1)

    args = parser.parse_args()
    for k in vars(args):
        print(f"{k.ljust(30)} = {args.__getattribute__(k)}")
    pl.seed_everything(args.seed)

    data_module = BioMRDataModule(args)
    data_module.prepare_data()
    # data_module.setup(stage="fit")

    trainer = pl.Trainer.from_argparse_args(
        args,
        # plugins=DDPPlugin(find_unused_parameters=False),
        callbacks=[
            EarlyStopping(monitor="acc/validation", min_delta=0.00, patience=3, verbose=False, mode="max"),
            ModelCheckpoint(
                monitor="acc/validation",
                filename="biomrc-e{epoch:02d}-{step}-acc{acc/validation:.4f}",
                save_top_k=3,
                mode="max",
                save_on_train_epoch_end=True,
                auto_insert_metric_name=False,
            )
        ],
        logger=TensorBoardLogger("logs", name=args.name, version=args.version, default_hp_metric=False) if args.train else None,
    )   

    if args.train:
        model = AOAReader_Model(args.learning_rate, args.embedding_dim, args.hidden_dim,
                    args.dropout_prob, args.seed, args.bert_learning_rate, args.bert_dir, args.flood,
                    args.occ_agg, args.tok_agg)
        trainer.fit(model, data_module)
    else:
        model = AOAReader_Model.load_from_checkpoint(args.test_ckpt_path)
    if args.test:
        result = trainer.test(model=model, datamodule=data_module)
        print(result)
    if args.predict:
        result = trainer.predict(model=model, datamodule=data_module)
        print(result)
        outputs = torch.nn.utils.rnn.pad_sequence([j.squeeze() for i in result for j in torch.split(i[1], 1) ], True)
        answers = torch.cat([i[0] for i in result])
        torch.save({"outputs": outputs, "answers": answers}, args.predict_file_name)
        embed()



if __name__ == "__main__":
    main()
