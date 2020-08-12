from pyknp import Juman
from torch import nn
import torch
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer, BertForSequenceClassification

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
        self.init_memory_vector = [torch.nn.Parameter(torch.zeros(1, 1, hidden, device = hparam.device))] * 2
        self.composer = nn.LSTM(input, hidden, num_layers=1, bidirectional=bidirectionality)
        self.composer.to(hparam.device)
    def init_hidden(self, batch_size):
        return (torch.cat(self.init_hidden_vector, dim=0), torch.cat(self.init_memory_vector, dim=0) )#if self.bidirectionality else self.init_hidden_vector[0]
    def forward(self, sentences):
        # Consider input (sentences) has form of (V,B,H)

        init_hidden = self.init_hidden(sentences.size(1))
        _, doc = self.composer(sentences.unsqueeze(1), init_hidden)
        return doc[0][0,0,:]
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
    # def forward(self, input_ids, masking):
    #     pass

class JumanAnalyzer(object):
    def __init__(self):
        self.jumanpp = Juman()
    def tokenize(self, sentence):
        # print(sentence)
        result = self.jumanpp.analysis(sentence)
        return [item.midasi for item in result.mrph_list()]
class BertMinMax(BaseClassifier):
    def __init__(self, path_to_bert, classes, hparam, langs):
        super(BertMinMax, self).__init__()
        self.hparams = hparam
        self.init_bert(path_to_bert, hparam.logger, hparam.device)
        # self.composer =
        # self.composer = RNNComposer(hparam=hparam, hidden=768, input=768, bidirectionality=True).to(hparam.device)
        # self.mixer = nn.Linear(768, 300)
        # self.mixer_rnn = nn.Linear(768*2,768).to(hparam.device)
        self.label = nn.ModuleDict({lang: nn.Linear(768*2, 12) for lang in langs})  #     Assuming BERT-BASE is used
        # for lang in self.label.keys():
        #     self.label[lang].to(hparam.device)
    def init_bert(self, path_to_bert, logger, device):
        self.model = BertModel.from_pretrained(path_to_bert)
        # self.model = BertForSequenceClassification.from_pretrained(path_to_bert)
        self.tokenizer = BertTokenizer.from_pretrained(path_to_bert)
        logger.info("Bert Model loaded")
        if not self.hparams.train_bert:
            for name, param in self.model.named_parameters():
                # param.requires_grad = False
                if name.startswith('embeddings'):
                    param.requires_grad = False
    def forward(self, input_ids, masking, lang):
        if not self.hparams.finetuning:
            cnt = 0
            last_hiddens = []
            while cnt< input_ids.size(0):
                last_hiddens = self.model(input_ids[cnt:cnt+25], masking[cnt:cnt+25])[0][:, 0, :]
                cnt+=25
            last_hiddens = torch.cat(last_hiddens, dim=0)
        else:
            last_hiddens = self.model(input_ids, masking)[0][:, 0, :]
        doc_representation = torch.cat([torch.mean(last_hiddens, dim=0), torch.max(last_hiddens, dim=0)[0]], dim=0)
        if self.hparams.decoder_sharing:
            topics = self.label[list(self.label.keys())[0]](doc_representation)
        else:
            topics = self.label[lang](doc_representation)
        topics_score = torch.sigmoid(topics)

        return topics_score
    def setoff_bert(self):
        if not self.hparams.train_bert:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if name.startswith('embeddings'):
                    param.requires_grad = False
    def token2id(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)
    def id2token(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent)

class BertRNN(BaseClassifier):
    def __init__(self, path_to_bert, classes, hparam, langs):
        super(BertRNN, self).__init__()
        self.hparams = hparam
        self.init_bert(path_to_bert, hparam.logger, hparam.device)
        # self.composer =
        # self.composer = RNNComposer(hparam=hparam, hidden=768, input=768, bidirectionality=True).to(hparam.device)
        # self.mixer = nn.Linear(768, 300)
        self.mixer_rnn = nn.ModuleDict({lang: RNNComposer(hparam=hparam, hidden=768, input=768, bidirectionality=True).to(hparam.device) for lang in langs})
        self.label = nn.ModuleDict({lang: nn.Linear(768, 12) for lang in langs})
            # {lang: nn.Linear(768, 12) for lang in langs}  #     Assuming BERT-BASE is used
        # for lang in self.label.keys():
        #     self.label[lang].to(hparam.device)
    def init_bert(self, path_to_bert, logger, device):
        if "distil" in path_to_bert:
            self.model = DistilBertModel.from_pretrained(path_to_bert)
            self.tokenizer = DistilBertModel.from_pretrained(path_to_bert)
        else:
            self.model = BertModel.from_pretrained(path_to_bert)
            # self.model = BertForSequenceClassification.from_pretrained(path_to_bert)
            self.tokenizer = BertTokenizer.from_pretrained(path_to_bert)
        logger.info("Bert Model loaded")
    def setoff_bert(self):
        if not self.hparams.train_bert:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if name.startswith('embeddings'):
                    param.requires_grad = False
    def forward(self, input_ids, masking, lang):
        if not self.hparams.finetuning:
            cnt = 0
            last_hiddens = []
            while cnt < input_ids.size(0):
                last_hiddens.append(self.model(input_ids[cnt:cnt + 100], masking[cnt:cnt + 100])[0][:, 0, :])
                cnt += 100
            last_hiddens = torch.cat(last_hiddens, dim=0)
        else:
            last_hiddens = self.model(input_ids, masking)[0][:, 0, :]
        if self.hparams.decoder_sharing:
            doc_representation = torch.cat([self.mixer_rnn[0](last_hiddens)], dim=0)
            topics = self.label[list(self.label.keys())[0]](doc_representation)
        else:
            doc_representation = torch.cat([self.mixer_rnn[lang](last_hiddens)], dim=0)
            topics = self.label[lang](doc_representation)
        topics_score = torch.sigmoid(topics)

        return topics_score

    def token2id(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)
    def id2token(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent)




