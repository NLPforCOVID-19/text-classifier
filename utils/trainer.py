import functools
import math
import statistics
from copy import deepcopy

from torch.utils.data import BatchSampler, RandomSampler
from tqdm import tqdm

import torch
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence
import torch.optim as optim
from enum import Enum
from .losses import f1_loss, xr_loss
from typing import List


class TrainMode(Enum):
    Train = 0,
    Dev = 1

def bysentence_finetuning(doc: List[torch.tensor], sample_size: int, device):
    doc_samples = [x[:64] for x in doc]
    doc_lens = [len(sent) for sent in doc_samples]
    doc_samples = pad_sequence(doc_samples[:75], batch_first=True)
    masking = torch.tensor([[0 if id < doc_lens[x] else 1 for id in range(doc_samples.size(1))] for x in range(doc_samples.size(0))], device=device)
    return doc_samples, masking
    pass
def bysentence(doc: List[torch.tensor], sample_size: int, device):
    doc_samples = doc
    doc_lens = [len(sent) for sent in doc_samples]
    doc_samples = pad_sequence(doc_samples, batch_first=True)
    masking = torch.tensor([[0 if id < doc_lens[x] else 1 for id in range(doc_samples.size(1))] for x in range(doc_samples.size(0))], device=device)
    return doc_samples, masking
    pass

def overlap_chunking(doc:List[torch.tensor], device: torch.device, chunk_len_limit: int, context_size:int, cls_token:int, sep_token:int):
    complete_article = torch.cat([x[1:-1] for x in doc], dim=0)
    samples = []
    cnt = 0
    while cnt<complete_article.size(0):
        samples.append(torch.tensor([cls_token]+ complete_article[cnt:cnt+chunk_len_limit].tolist() + [sep_token], device=device))
        if cnt+chunk_len_limit>=complete_article.size(0):
            cnt+=chunk_len_limit
        else:
            cnt+=chunk_len_limit-context_size
    max_len = max([x.size(0) for x in samples])
    masking = torch.tensor(
        [[0 if id < x.size(0) else 1 for id in range(max_len)] for x in samples],
        device=device)
    return pad_sequence(samples, batch_first=True), masking

def overlap_chunking_finetuning(doc:List[torch.tensor], device: torch.device, chunk_len_limit: int, context_size:int, cls_token:int, sep_token:int):
    complete_article = torch.cat([x[1:-1] for x in doc], dim=0)
    samples = []
    cnt = 0
    seq_cnt = 0
    while cnt<complete_article.size(0) and seq_cnt<50:
        samples.append(torch.tensor([cls_token]+ complete_article[cnt:cnt+chunk_len_limit].tolist() + [sep_token], device=device))
        if cnt+chunk_len_limit>=complete_article.size(0):
            cnt+=chunk_len_limit
        else:
            cnt+=chunk_len_limit-context_size
        seq_cnt+=1
    max_len = max([x.size(0) for x in samples])
    masking = torch.tensor(
        [[0 if id < x.size(0) else 1 for id in range(max_len)] for x in samples],
        device=device)
    return pad_sequence(samples, batch_first=True), masking



class Trainer(object):
    def __init__(self, hparams, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.hparams = hparams
        self.model = model
        # self.criterion_ce = nn.BCELoss().to(hparams.device)
        self.criterion_xr = xr_loss
        # self.criterion = nn.BCELoss().to(hparams.device) if self.hparams.use_ce else f1_loss
        # self.criterion_rmse = nn.MSELoss().to(hparams.device)
        self.optimizer = optimizer
        self.device = hparams.device
        self.epoch = 0
        self.teacher_forcing_ratio = 0.5
class SentenceTrainer(Trainer):
    def __init__(self, hparams, model, criterion, optimizer):
        super(SentenceTrainer, self).__init__( hparams, model, criterion, optimizer)

    def train(self, dataset, mode=TrainMode.Train):
        if mode == TrainMode.Train:
            self.model.train()
            self.optimizer.zero_grad()
        total_loss_ce, total_loss_rmse = 0.0, 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device=self.device)
        test_loss_ce, test_loss_rmse = 0.0, 0.0
        self.criterion_ce = nn.BCELoss(weight=dataset.pos_weight).to(self.hparams.device)
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            doc, tags, lang = dataset[indices[idx]]
            if self.hparams.train_by_chunk and self.hparams.finetuning:
                doc_sample, masking = overlap_chunking_finetuning(doc, self.device, self.hparams.chunk_len_limit,
                                                                      self.hparams.context_size,
                                                                      self.model.tokenizer.cls_token_id,
                                                                      self.model.tokenizer.sep_token_id)
            elif self.hparams.train_by_chunk and not self.hparams.finetuning:
                doc_sample, masking = overlap_chunking(doc, self.device, self.hparams.chunk_len_limit,
                                                           self.hparams.context_size,
                                                           self.model.tokenizer.cls_token_id,
                                                           self.model.tokenizer.sep_token_id)
            elif not self.hparams.finetuning:
                doc_sample, masking = bysentence(doc, self.hparams.sample_size, self.device)
            else:
                doc_sample, masking = bysentence_finetuning(doc, self.hparams.sample_size, self.device)
            result = self.model(doc_sample.type(torch.long), masking, lang, indices[idx])
            loss_ce = self.criterion_ce(result, tags)
            loss_xr = self.criterion_xr(result, tags)
            if self.hparams.combine_xr:
                loss = loss_ce + loss_xr
            else:
                loss = loss_ce
            total_loss_ce += loss.item()
            test_loss_ce += loss.item()
            loss.backward()
            if (idx + 1) % self.hparams.batchsize == 0:  # pesudo minibatch
                self.optimizer.step()
                self.optimizer.zero_grad()
            if (idx + 1) % self.hparams.error_report == 0:
                self.hparams.writer.add_scalar(
                    'Loss/train',
                    test_loss_ce / self.hparams.error_report,
                    self.epoch * len(dataset) + idx
                )
                test_loss_ce = 0
        return (total_loss_rmse+total_loss_ce) / len(dataset)

    def eval(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss_ce, total_loss_rmse = 0.0, 0.0
            test_loss_ce, test_loss_rmse = 0.0, 0.0
            for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
                doc, tags, lang = dataset[idx]

                for _ in range(1):#range(self.hparams.sample_count if self.hparams.finetuning else 1):
                    if self.hparams.train_by_chunk and self.hparams.finetuning:
                        doc_sample, masking = overlap_chunking_finetuning(doc, self.device, self.hparams.chunk_len_limit,
                                                               self.hparams.context_size,
                                                               self.model.tokenizer.cls_token_id,
                                                               self.model.tokenizer.sep_token_id)
                    elif self.hparams.train_by_chunk and not self.hparams.finetuning:
                        doc_sample, masking = overlap_chunking(doc, self.device, self.hparams.chunk_len_limit,
                                                               self.hparams.context_size,
                                                               self.model.tokenizer.cls_token_id,
                                                               self.model.tokenizer.sep_token_id)
                    elif not self.hparams.finetuning:
                        doc_sample, masking = bysentence(doc, self.hparams.sample_size, self.device)
                    else:
                        doc_sample, masking = bysentence_finetuning(doc, self.hparams.sample_size, self.device)
                    # try:
                    result = self.model(doc_sample.type(torch.long), masking, lang, -1)
                    loss_ce = self.criterion_ce(result, tags)
                    loss_xr = self.criterion_xr(result, tags)
                    if self.hparams.combine_xr:
                        loss = loss_ce + loss_xr
                    else:
                        loss = loss_ce
                    total_loss_ce += loss.item()
                    test_loss_ce += loss.item()
                if (idx + 1) % self.hparams.error_report == 0:
                    self.hparams.writer.add_scalar(
                        'Loss/eval',
                        test_loss_ce / self.hparams.error_report,
                        self.epoch * len(dataset) + idx
                    )
                    test_loss_ce = 0
        return (total_loss_rmse+total_loss_ce) / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            results = []
            for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
                doc, tags, lang = dataset[idx]
                # isRelated, clarity, usefulness, topics = tags
                for _ in range(1):
                    if self.hparams.train_by_chunk and self.hparams.finetuning:
                        doc_sample, masking = overlap_chunking_finetuning(doc, self.device,
                                                                          self.hparams.chunk_len_limit,
                                                                          self.hparams.context_size,
                                                                          self.model.tokenizer.cls_token_id,
                                                                          self.model.tokenizer.sep_token_id)
                    elif self.hparams.train_by_chunk and not self.hparams.finetuning:
                        doc_sample, masking = overlap_chunking(doc, self.device, self.hparams.chunk_len_limit,
                                                               self.hparams.context_size,
                                                               self.model.tokenizer.cls_token_id,
                                                               self.model.tokenizer.sep_token_id)
                    elif not self.hparams.finetuning:
                        doc_sample, masking = bysentence(doc, self.hparams.sample_size, self.device)
                    else:
                        doc_sample, masking = bysentence_finetuning(doc, self.hparams.sample_size, self.device)
                    predictions = self.model(doc_sample.type(torch.long), masking, lang,  -1)
                    results.append(predictions)
        return torch.stack(results, dim=0)
