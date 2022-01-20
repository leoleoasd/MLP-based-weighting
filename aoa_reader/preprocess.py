import json
import string

from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

from IPython import embed
import re

def process(input, output):
    with open(input) as f:
        data = json.load(f)
    new = []
    for i in tqdm(range(len(data['abstracts']))):
        # embed()
        entity_map = {}
        tt = []
        for x in data['entities_list'][i]:
            v = eval(x.split(" :: ")[-1])[0]
            tt.append(v)
            entity_map[x.split(" :: ")[0]] = v
            if type(v) != str:
                print("Error!")
                embed()
        abstract = data['abstracts'][i].lower()
        title = data['titles'][i].lower()
        for k in entity_map:
            try:
                abstract = re.sub(repr(k)[1:-1] + r"\b", repr(entity_map[k])[1:-1], abstract)#abstract.replace(k + " ", entity_map[k] + " ")
            except Exception as e:
                print(k)
                print(k)
                print(repr(k))
                print(entity_map)
                print(data['entities_list'][i])
                exit(0)
                raise e
            title = re.sub(repr(k)[1:-1] + r"\b", repr(entity_map[k])[1:-1], title)
        new.append({
            'abstract': abstract,
            'title': title.replace('.xxxx', ' [MASK]').replace('xxxx', '[MASK]'),
            'entities_list': tt,
            'answer': tt.index(entity_map[data['answers'][i].split(" :: ")[0]])
        })
    with open(output, "w") as f:
        json.dump(new, f)
