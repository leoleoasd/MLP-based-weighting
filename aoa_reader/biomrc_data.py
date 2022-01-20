import json
import os
import pickle
from os import path

import pytorch_lightning as pl
import torch
import tqdm
from transformers import BertTokenizer
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
# from nltk import sent_tokenize

from .preprocess import process
from IPython import embed


class BioMRCDataset(torch.utils.data.IterableDataset):
    def __init__(self, files, tokenizer: BertTokenizer):
        super().__init__()
        self.files = files
        self.tokenizer = tokenizer

    def tokenize(self, i):
        data = {}
        abstract = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(
                "[CLS] " + i['abstract'] + " [SEP]"
            )
        )
        # bert max_seq_length
        if len(abstract) > 512:
            return None
        data['abstract'] = torch.tensor(abstract)
        data['title'] = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(
                    "[CLS] " + i['title'] + " [SEP]"
                )
            )
        )
        # only use the first token of entity.
        data['entities_list'] = torch.tensor([
            self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(
                    e
                )
            )[0]
            for e in i['entities_list']
        ])
        data['answer'] = i['answer']
        return data

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            for file in self.files:
                with open(file) as f:
                    d = json.load(f)
                yield from d
                # for i in d:
                #     t = self.tokenize(i)
                #     if t is not None:
                #         yield t
        else:
            count = 0
            for file in self.files:
                with open(file) as f:
                    d = json.load(f)
                for i in d:
                    if count % worker_info.num_workers == worker_info.id:
                        yield i
                        # t = self.tokenize(i)
                        # if t is not None:
                        #     yield t
                    count += 1


class BioMRDataModule(pl.LightningDataModule):
    url = "https://archive.org/download/biomrc_dataset/biomrc_splitted/biomrc_{data_size}.tar.gz"
    md5 = {
        'large': '2810ba8bd17ce064a750d8b42ecfe4c4',
        'small': '568e91f0de5aff7521f83e16491104fb',
        'tiny': '0b4b60a15523311d8b0925112f6c9c52'
    }

    def __init__(self, args):
        super().__init__()
        self.bert_dir = args.bert_dir
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        # self.vocab = None
        self.tokenizer = None
        # self.vocab_size = 0
        self.data_size = args.data_size
        self.raw_folder = args.raw_folder
        self.processed_data_dir = args.processed_data_dir
        self.batch_size = args.batch_size
        # self.vocab_dir = args.vocab_dir
        self.num_workers = args.dataloader_num_workers
        self.context_threshold = args.context_threshold
        self.pred_data = args.pred_data

    def gen_raw_file_name(self):
        if self.data_size == 'tiny':
            return [f"{self.raw_folder}/dataset_{self.data_size}.json"]
        elif self.data_size == 'large':
            return [f"{self.raw_folder}/dataset_part{i}.json" for i in range(1, 10)]
        else:
            return [f"{self.raw_folder}/dataset_part{i}_{self.data_size}.json" for i in range(1, 10)]

    def gen_processed_file_name(self, stage = "all"):
        if self.data_size == 'tiny':
            if stage == "all":
                return [f"{self.processed_data_dir}/dataset_{self.data_size}.json"]
            elif stage == "train":
                return []
            elif stage == "val":
                return []
            elif stage == "test":
                return [f"{self.processed_data_dir}/dataset_{self.data_size}.json"]
            else:
                raise Exception("invalid stage")
        elif self.data_size == 'large':
            files = [f"{self.processed_data_dir}/dataset_part{i}.json" for i in range(1, 10)]
            if stage == "all":
                return files
            elif stage == "train":
                return files[0:7]
            elif stage == "val":
                return [files[7]]
            elif stage == "test":
                return [files[8]]
            else:
                raise Exception("invalid stage")
        else:
            files = [f"{self.processed_data_dir}/dataset_part{i}_{self.data_size}.json" for i in range(1, 10)]
            if stage == "all":
                return files
            elif stage == "train":
                return files[0:7]
            elif stage == "val":
                return [files[7]]
            elif stage == "test":
                return [files[8]]
            else:
                raise Exception("invalid stage")

    def prepare_data(self, stage=None):
        if os.path.isfile(self.gen_processed_file_name("all")[0]):
            return
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        print("Downloading {}".format(self.url.format(data_size=self.data_size)))
        url = self.url.format(data_size=self.data_size)
        filename = f"biomrc_{self.data_size}.tar.gz"
        download_and_extract_archive(
            url,
            download_root=self.raw_folder,
            filename=filename,
            md5=self.md5[self.data_size]
        )
        # pre-process file.
        print("pre-processing")
        for raw, pro in tqdm.tqdm(zip(self.gen_raw_file_name(), self.gen_processed_file_name())):
            process(
               raw, pro
            )

    def setup(self, stage=None):
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_dir)
        self.train_dataset = BioMRCDataset(
            self.gen_processed_file_name("train"), self.tokenizer)
        self.val_dataset = BioMRCDataset(
            self.gen_processed_file_name("val"), self.tokenizer)
        self.test_dataset = BioMRCDataset(
            self.gen_processed_file_name("test"), self.tokenizer)
        if self.pred_data == -1:
            self.predict_dataset = BioMRCDataset(
                self.gen_processed_file_name("all"), self.tokenizer)
        else:
            self.predict_dataset = BioMRCDataset(
                [self.gen_processed_file_name("all")[self.pred_data]], self.tokenizer)


    def collate(self):
        def _collate(batch):
            abstracts = []
            titles = []
            entities_list = []
            answers = []
            max_ent_count = 0
            max_cand_length = 0
            for i in batch:
                abstracts.append(i['abstract'])
                titles.append(i['title'])
                # only use the first token of entity.
                entities_list.append([torch.tensor(
                    self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(
                            e
                        )
                    )
                ) for e in i['entities_list'] ])
                entities_list[-1] = pad_sequence(entities_list[-1], True)
                max_ent_count = max(max_ent_count, entities_list[-1].size(0))
                max_cand_length = max(max_cand_length, entities_list[-1].size(1))
                answers.append(i['answer'])
            
            for i in range(len(entities_list)):
                entities_list[i] = F.pad(entities_list[i], (0, max_cand_length - entities_list[i].size(1), 0, max_ent_count - entities_list[i].size(0)))

            entities_list = torch.stack(entities_list)

            tokenized = self.tokenizer(text=abstracts,
                                       text_pair=titles,
                                       return_tensors="pt",
                                       max_length=512,
                                       padding=True,
                                       truncation="only_first")
            # input_ids = tokenized.input_ids.cuda()
            # attention_mask = tokenized.attention_mask.cuda()
            # token_type_ids = tokenized.token_type_ids.cuda()

            context = tokenized.input_ids.masked_fill(
                torch.logical_and(
                    tokenized.attention_mask, tokenized.token_type_ids), 0)
            question = tokenized.input_ids.masked_fill(
                torch.logical_and(
                    tokenized.attention_mask, 1 - tokenized.token_type_ids), 0)

            # abstracts = pad_sequence(abstracts, True)
            # titles = pad_sequence(titles, True)
            # entities_list = pad_sequence(entities_list, True)
            return (tokenized.input_ids,
                    tokenized.attention_mask,
                    tokenized.token_type_ids,
                    context,
                    question,
                    entities_list,
                    torch.tensor(answers))
        return _collate

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate(),
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate(),
            batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate(),
            batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            collate_fn=self.collate(),
            batch_size=self.batch_size,
            persistent_workers=True,
        )
