from torch import nn
import torch
from transformers import BertModel, BertTokenizer

class SentenceComposer(nn.Module):
    def __init__(self):
        super(SentenceComposer, self).__init__()
class RNNComposer(SentenceComposer):
    def __init__(self, hidden, input, hparam, bidirectionality: bool = True):
        super(RNNComposer, self).__init__()
        self.bidirectionality = bidirectionality
        self.hidden = hidden
        self.init_hidden = torch.zeros(1*bidirectionality, 1, hidden, device = hparam.device)
        self.composer = nn.GRU(input, hidden, num_layer=1, bidirectional=bidirectionality)
        self.composer.to(hparam.device)
    def init_hidden(self, batch):
        return torch.concat(self.init_hidden, dim=0) if self.bidirectionality else self.init_hidden[0]
    def forward(self, sentences):
        # Consider input (sentences) has form of (V,B,H)
        init_hidden = self.init_hidden(sentences.size(1))
        _, doc = self.composer(sentences, init_hidden)
        return doc
class BertClassifier(nn.Module):
    def __init__(self, path_to_bert, classes, hparam):
        super(BertClassifier, self).__init__()
        self.model, self.tokenizer = BertModel, BertTokenizer
        self.init_bert(path_to_bert)
        self.composer = RNNComposer(hparam=hparam, hidden=768, input=768, bidirectionality=True)
        self.labels = nn.Linear(768, classes)  # Assuming BERT-BASE is used
        self.is_related = nn.Linear(768, 1)
        self.usefulness = nn.Linear(768, 1)
        self.clarity = nn.Linear(768, 1)

    def init_bert(self, path_to_bert):
        self.model.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
        self.tokenizer.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
    def forward(self, input_ids):

        #input_ids should be organized as (V, ids), ids only contains one sentence -> which means only one document at a time

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


