from pyknp import Juman
from torch import nn
import torch
from transformers import BertModel, BertTokenizer

from model.peripheral import Attn, AttnType


class SentenceComposer(nn.Module):
    def __init__(self):
        super(SentenceComposer, self).__init__()
class RNNComposer(SentenceComposer):
    def __init__(self, hidden, input, hparam, bidirectionality: bool = True):
        super(RNNComposer, self).__init__()
        self.bidirectionality = bidirectionality
        self.hidden = hidden
        self.init_hidden_vector = [torch.nn.Parameter(torch.zeros(1, 1, hidden, device = hparam.device)) ] *2
        self.composer = nn.GRU(input, hidden, num_layers=1, bidirectional=bidirectionality)
        self.composer.to(hparam.device)
    def init_hidden(self, batch_size):
        return torch.cat(self.init_hidden_vector, dim=0) #if self.bidirectionality else self.init_hidden_vector[0]
    def forward(self, sentences):
        # Consider input (sentences) has form of (V,B,H)
        init_hidden = self.init_hidden(sentences.size(1))
        _, doc = self.composer(sentences, init_hidden)
        return doc
class CNNComposer(SentenceComposer):
    def __init__(self, hidden, input, hparam):
        super(CNNComposer, self).__init__()
        # self.bidirectionality = bidirectionality
        # self.hidden = hidden
        # self.init_hidden_vector = [torch.nn.Parameter(torch.zeros(1, 1, hidden, device = hparam.device)) ] *2
        # self.composer = nn.GRU(input, hidden, num_layers=1, bidirectional=bidirectionality)
        self.conv1 = nn.Conv1d(768, 768, 4, stride=2)
        self.pool1 = nn.MaxPool1d(3, stride=2)
        self.conv1 = nn.Conv1d(768, 768, 3, stride=2)
        self.pool1 = nn.MaxPool1d(3, stride=2)
        self.conv1 = nn.Conv1d(768, 768, 3, stride=2)
    # def init_hidden(self, batch_size):
    #     return torch.cat(self.init_hidden_vector, dim=0) #if self.bidirectionality else self.init_hidden_vector[0]
    def forward(self, sentences):
        # Consider input (sentences) has form of (V,B,H)
        init_hidden = self.init_hidden(sentences.size(1))
        _, doc = self.composer(sentences, init_hidden)
        return doc

class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()
        self.sentence_extractor = None
        self.doc_extractor = None
        self.classifier = None
    def forward(self, input_ids):
        pass

class JumanAnalyzer(object):
    def __init__(self):
        self.jumanpp = Juman()
    def tokenize(self, sentence):
        # print(sentence)
        result = self.jumanpp.analysis(sentence)
        return [item.midasi for item in result.mrph_list()]
class BertClassifier(BaseClassifier):
    def __init__(self, path_to_bert, classes, hparam):
        super(BertClassifier, self).__init__()
        self.hparams = hparam
        self.init_bert(path_to_bert, hparam.logger, hparam.device)
        # self.composer =
        self.composer = [RNNComposer(hparam=hparam, hidden=768, input=768, bidirectionality=True).to(hparam.device) for _ in range(4)]
        # self.mixer = nn.Linear(768, 300)
        self.mixer_rnn = [nn.Linear(768*2,768).to(hparam.device) for _ in range(4)]
        self.labels = nn.Linear(768, 1)  # Assuming BERT-BASE is used
        self.is_related = nn.Linear(768, 1)
        self.usefulness = nn.Linear(768, 1)
        self.clarity = nn.Linear(768, 1)
        self.attn = [Attn(AttnType.general, 768, 768).to(hparam.device) for _ in range(4)]
    def setoff_composer(self):
        for composer in self.composer:
            for param in composer.composer.parameters():
                param.requires_grad=False
    def setoff_linear(self):
        for param in self.labels.parameters():
            param.requires_grad=False
        for param in self.usefulness.parameters():
            param.requires_grad=False
        for param in self.clarity.parameters():
            param.requires_grad=False
        for param in self.is_related.parameters():
            param.requires_grad=False
        for attn in self.attn:
            for param in attn.fh.parameters():
                param.requires_grad = False
    def seton_everything(self):
        for composer in self.composer:
            for param in composer.composer.parameters():
                param.requires_grad=True
        for param in self.labels.parameters():
            param.requires_grad=True
        for param in self.usefulness.parameters():
            param.requires_grad=True
        for param in self.clarity.parameters():
            param.requires_grad=True
        for param in self.is_related.parameters():
            param.requires_grad=True
        for attn in self.attn:
            for param in attn.fh.parameters():
                param.requires_grad = True


    def init_bert(self, path_to_bert, logger, device):
        self.model = BertModel.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
        self.tokenizer = BertTokenizer.from_pretrained(path_to_bert, cache_dir=None, from_tf=False, state_dict=None)
        logger.info("Bert Model loaded")
        if not self.hparams.train_bert:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                # if name.startswith('embeddings'):
                #     param.requires_grad = False
    def forward(self, input_ids):

        #input_ids should be organized as (V, ids), ids only contains one sentence -> which means only one document at a time
        cnt = 0
        last_hiddens = []
        doc_len = input_ids.size(0)

        # print(input_ids)
# /

        last_hidden = self.model(input_ids)[0]
        output_hidden = [None for _ in range(4)]
        doc_representation = [None for _ in range(4)]
        for i in range(4):
            output_hidden[i] = self.composer[i](last_hidden[:, 0, :].unsqueeze(1))
            output_hidden[i] = torch.cat((output_hidden[i][0], output_hidden[i][1]), dim=1)
            # print(output_hidden[i].size())

            doc_representation[i] = self.attn[i](self.mixer_rnn[i](output_hidden[i].view(-1)), last_hidden[:, 0, :])


        # output_hidden_relateness = self.composer[i](last_hidden[:, 0, :].unsqueeze(1))  #B,V,H
        # output_hidden_clearness = self.composer(last_hidden[:, 0, :].unsqueeze(1))
        # output_hidden_clearness = self.composer(last_hidden[:, 0, :].unsqueeze(1))
        # output_hidden_other = self.composer(last_hidden[:, 0, :].unsqueeze(1))


            # print(torch.cat((output_hidden[0], output_hidden[1]), dim=1))

            # print(last_hidden)



        # doc_representation = self.mixer(doc_representation)



        is_related = self.is_related(doc_representation[0])
        is_related_score = torch.sigmoid(is_related)

        clarity = self.clarity(doc_representation[1])
        clarity_score = torch.sigmoid(clarity)

        usefulness = self.usefulness(doc_representation[2])
        usefulness_score = torch.sigmoid(usefulness)

        topics = self.labels(doc_representation[3])
        topics_score = torch.sigmoid(topics)

        return torch.cat((is_related_score,  topics_score, clarity_score, usefulness_score), dim=0)#, torch.cat((clarity_score, usefulness_score), dim=0)

    def token2id(self, tokens):
        # Input tokens are a list of token
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)
    def id2token(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent)




