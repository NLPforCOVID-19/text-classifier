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
from .losses import f1_loss
from typing import List


class TrainMode(Enum):
    Train = 0,
    Dev = 1

def sampling(doc: List[List[str]], sample_size: int, device, finetuning: bool):
    if finetuning:
        title = doc[0]
        doc = doc[1:]
        sample_idx = torch.randperm(len(doc))
        sample_idx = sorted(sample_idx[:sample_size])
        doc_samples = [title] + [doc[id] for id in sample_idx[:sample_size]]
        # doc_samples, doc_lens = pack_sequence(doc_samples, enforce_sorted=False)
        doc_lens = [len(sent) for sent in doc_samples]
    else:
        doc_samples = doc
        # doc_samples, doc_lens = pack_sequence(doc_samples, enforce_sorted=False)
        doc_lens = [len(sent) for sent in doc_samples]
    doc_samples = pad_sequence(doc_samples, batch_first=True)
    masking = torch.tensor([[0 if id < doc_lens[x] else 1 for id in range(doc_samples.size(1))] for x in range(doc_samples.size(0))], device=device)
    return doc_samples, masking
    pass

class Trainer(object):
    def __init__(self, hparams, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.hparams = hparams
        self.model = model
        self.criterion = nn.BCELoss().to(hparams.device) if self.hparams.use_ce else f1_loss
        self.criterion_rmse = nn.MSELoss().to(hparams.device)
        self.optimizer = optimizer
        self.device = hparams.device
        self.epoch = 0
        self.teacher_forcing_ratio = 0.5
    def train(self, dataset, mode=TrainMode.Train):
        if mode == TrainMode.Train:
            self.model.train()
            self.optimizer.zero_grad()
        total_loss_ce, total_loss_rmse = 0.0, 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device=self.device)
        test_loss_ce, test_loss_rmse = 0.0, 0.0

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            doc, tags, lang = dataset[indices[idx]]
            for _ in range(self.hparams.sample_count if self.hparams.finetuning else 1):
                doc_sample, masking = sampling(doc, self.hparams.sample_size, self.device, self.hparams.finetuning)
                # print(doc_sample.size())
                result = self.model(doc_sample.type(torch.long), masking, lang)
                loss_ce = self.criterion(result, tags)
                total_loss_ce += loss_ce.item()
                test_loss_ce += loss_ce.item()
                loss_ce.backward()
            if (idx + 1) % self.hparams.batchsize == 0:  # pesudo minibatch
                self.optimizer.step()
                self.optimizer.zero_grad()
            if (idx + 1) % self.hparams.error_report == 0:
                self.hparams.logger.info(
                    'Epoch {}, Batch {} : Avg CE Loss {}'.format(self.epoch, idx, test_loss_ce / (
                                self.hparams.sample_count * self.hparams.error_report if self.hparams.finetuning else  self.hparams.error_report)))
                test_loss_ce = 0
        return (total_loss_rmse+total_loss_ce) / len(dataset)

    def eval(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss_ce, total_loss_rmse = 0.0, 0.0
            test_loss_ce, test_loss_rmse = 0.0, 0.0
            for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
                doc, tags, lang = dataset[idx]

                for _ in range(self.hparams.sample_count if self.hparams.finetuning else 1):
                    doc_sample, masking = sampling(doc, self.hparams.sample_size, self.device, self.hparams.finetuning)
                    # try:
                    result = self.model(doc_sample.type(torch.long), masking, lang)
                    loss_ce = self.criterion(result, tags)
                    total_loss_ce += loss_ce.item()
                    test_loss_ce += loss_ce.item()
                if (idx + 1) % self.hparams.error_report == 0:
                    self.hparams.logger.info(
                        'Epoch {}, Batch {} : Avg CE Loss {}'.format(self.epoch, idx,
                                                                     test_loss_ce / (
                                                                                 self.hparams.sample_count * self.hparams.error_report if self.hparams.finetuning else  self.hparams.error_report)))
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
                    doc_sample, masking = sampling(doc, self.hparams.sample_size, self.device, self.hparams.finetuning)
                    predictions = self.model(doc_sample.type(torch.long), masking, lang)
                    results.append(predictions)
        return torch.stack(results, dim=0)
