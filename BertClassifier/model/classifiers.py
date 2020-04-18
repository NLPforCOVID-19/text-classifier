from torch import nn
import torch
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, path_to_bert):
        super(BertClassifier, self).__init__()
        self.model, self.tokenizer = BertModel, BertTokenizer
        self.init_bert(path_to_bert)
        self.labels = [nn.Linear()]
    def init_bert(self, path_to_bert):
        self.model.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
        self.tokenizer.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
    def forward(self, input_ids):
        last_hidden = self.model(input_ids)[0]
        



