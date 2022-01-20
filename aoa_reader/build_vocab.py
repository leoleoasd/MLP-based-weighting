import json
import pickle

from torchtext import vocab
from tqdm import tqdm


def build_vocab(files):
    def iter():
        for file in tqdm(files):
            with open(file) as f:
                d = json.load(f)
            for i in tqdm(d):
                yield i['abstracts']
                yield i['titles']

    return vocab.build_vocab_from_iterator(iter().__iter__(), specials=['<PAD>']).get_stoi()

    # with open("vocab.dat", "wb") as f:
    #     pickle.dump(voc.get_stoi(), f)
