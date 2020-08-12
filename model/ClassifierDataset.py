import re
from copy import deepcopy

import torch
import torch.utils.data as data
import json


class SimpleVocab(object):
    def __init__(self, items):
        cnt = 0
        self.token2Idx = {}
        self.idx2Topic = {}
        for topic in items:
            self.token2Idx[topic] = cnt
            self.idx2Topic[cnt] = topic
            cnt += 1
    def convertIdx2Token(self, idx):
        return torch.tensor(self.idx2Topic[idx])
    def convertTokens2Idxs(self, tokens):
        return torch.stack([self.convertToken2Idx(token) for token in tokens], dim=0)
    def convertToken2Idx(self, token):
        return torch.tensor(self.topic2Idx[token])



class BCDataset(data.Dataset):
    def __init__(self, docs, tags, doc_langs):
        super(BCDataset, self).__init__()
        self.docs = docs
        self.tags = tags
        self.size = len(docs)
        self.doc_langs = doc_langs
    def __len__(self):
        return self.size
    def __getitem__(self, index): #Assuming the input index is a group
        if not isinstance(index, list):
            doc = deepcopy(self.docs[index])
            tag = deepcopy(self.tags[index])
            lang = deepcopy(self.doc_langs[index])
        else:
            doc = [deepcopy(self.docs[id]) for id in index]
            tag = deepcopy(self.tags[index])
        return doc, tag, lang
    def get_langs(self):
        return set(self.doc_langs)
    @staticmethod
    def loadData(file, tokenize, convertToken2Idx, device):
        with open(file, encoding='utf-8') as f:
            items = f.readlines()
            items = [json.loads(item) for item in items]
        docs = []
        doc_langs = []
        tags = []
        for item in items:
            # print(item)
            sents = item["cleaned_text"]
            sents = re.split('; |\*|\n|\. |。',sents)
            sents = [tokenize(item["title"])] + [tokenize(x.strip('\n')) for x in sents]
            sents = list(filter(lambda x: len(x) > 5, sents))
            sents = [x[:128] for x in sents]
            sents = [torch.tensor(convertToken2Idx(['CLS'] + x + ['SEP']), dtype=torch.long, device = device) for x in sents]

            if len(sents) == 0:
                continue
            docs.append(sents)
            doc_langs.append(item["lang"])
            tag = ([item["tags"]["is_about_COVID-19"]]+ [min(1, item["tags"]["is_useful"])]+
                                       [item["tags"]["is_clear"]]+ [item["tags"]["is_about_false_rumor"]]+
                                       [tuple[1] for tuple in item["tags"]["topics"].items()])
            tags.append(torch.tensor(tag, dtype=torch.float, device = device))

        return BCDataset(docs, tags, doc_langs)


