from pyknp import Juman
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
        self.init_hidden_vector = [torch.zeros(1, 1, hidden, device = hparam.device) ] *2
        self.composer = nn.GRU(input, hidden, num_layers=1, bidirectional=bidirectionality)
        self.composer.to(hparam.device)
    def init_hidden(self, batch_size):
        return torch.cat(self.init_hidden_vector, dim=0) #if self.bidirectionality else self.init_hidden_vector[0]
    def forward(self, sentences):
        # Consider input (sentences) has form of (V,B,H)
        init_hidden = self.init_hidden(sentences.size(1))
        _, doc = self.composer(sentences, init_hidden)
        return doc

class JumanAnalyzer(object):
    def __init__(self):
        self.jumanpp = Juman()
    def tokenize(self, sentence):
        # print(sentence)
        result = self.jumanpp.analysis(sentence)
        return [item.midasi for item in result.mrph_list()]
class BertClassifier(nn.Module):
    def __init__(self, path_to_bert, classes, hparam):
        super(BertClassifier, self).__init__()
        self.hparams = hparam
        self.init_bert(path_to_bert, hparam.logger, hparam.device)
        self.composer = RNNComposer(hparam=hparam, hidden=300, input=768, bidirectionality=True).to(hparam.device)
        self.labels = nn.Linear(300, classes)  # Assuming BERT-BASE is used
        self.is_related = nn.Linear(300, 1)
        self.usefulness = nn.Linear(300, 1)
        self.clarity = nn.Linear(300, 1)

    def init_bert(self, path_to_bert, logger, device):
        self.model = BertModel.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
        self.model.to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
        logger.info("Bert Model loaded")
    def forward(self, input_ids):

        #input_ids should be organized as (V, ids), ids only contains one sentence -> which means only one document at a time
        cnt = 0
        last_hiddens = []
        doc_len = input_ids.size(0)
        # while cnt < doc_len:
        #     if cnt+25 < doc_len:
        #         last_hiddens.append(self.model(input_ids[cnt:cnt+25])[0])
        #     else:
        #         last_hiddens.append(self.model(input_ids[cnt:doc_len])[0])
        #     cnt+=25
        # last_hidden = torch.cat(last_hiddens, dim=0)

        last_hidden = self.model(input_ids)[0]

        doc_representation = self.composer(last_hidden[:, 0, :].unsqueeze(1))

        doc_representation = (doc_representation[0] + doc_representation[1]).view(-1)


        is_related = self.is_related(doc_representation)
        is_related_score = torch.sigmoid(is_related)

        clarity = self.clarity(doc_representation)
        clarity_score = torch.relu(clarity)

        usefulness = self.usefulness(doc_representation)
        usefulness_score = torch.relu(usefulness)

        topics = self.labels(doc_representation)
        topics_score = torch.sigmoid(topics)

        print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
        print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
        print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))

        return torch.cat((is_related_score,  topics_score), dim=0), torch.cat((clarity_score, usefulness_score), dim=0)

    def token2id(self, tokens):
        # Input tokens are a list of token
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)
    def id2token(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)



