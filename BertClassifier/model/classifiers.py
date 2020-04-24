from torch import nn
import torch
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, path_to_bert, classes):
        super(BertClassifier, self).__init__()
        self.model, self.tokenizer = BertModel, BertTokenizer
        self.init_bert(path_to_bert)
        self.labels = nn.Linear(768, classes)  # Assuming BERT-BASE is used
        self.is_related = nn.Linear(768, 1)
        self.usefulness = nn.Linear(768, 1)
        self.clarity = nn.Linear(768, 1)

    def init_bert(self, path_to_bert):
        self.model.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
        self.tokenizer.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
    def forward(self, input_ids):
        last_hidden = self.model(input_ids)[0]

        labels_logit = self.labels(last_hidden)
        labels_score = nn.LogSigmoid(labels_logit)

        is_related = self.is_related(last_hidden)
        is_related_score = nn.LogSigmoid(is_related)

        clarity = self.clarity(last_hidden)
        clarity_score = nn.LogSigmoid(clarity)

        return (labels_score, is_related_score, clarity_score)

    def token2id(self, tokens):
        # Input tokens are a list of token
        return self.tokenizer.convert_tokens_to_ids(tokens)
    def id2token(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)


