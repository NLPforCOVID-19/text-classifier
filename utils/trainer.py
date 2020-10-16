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

class TrainMode(Enum):
    Train = 0,
    Dev = 1

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
        if self.epoch % 2 == 1:
            self.model.seton_everything()
            self.model.setoff_composer()
        else:
            self.model.seton_everything()
            self.model.setoff_linear()
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            doc, tags = dataset[indices[idx]]
            # doc = [sent[:64] for sent in doc]
            # doc = doc
            # doc = doc[:96]
            isRelated,isRoumer, clarity, usefulness, topics = tags
            # print(topics.size())
            doc = pad_sequence(doc).transpose(0,1) # padding document
            #result_ce, result_rmse = self.model(doc)
            result = self.model(doc)
            # print(result)
            #loss_ce = self.criterion(result_ce, torch.cat((isRelated, topics), dim=0))
            # print(result)
            loss_ce = self.criterion(result, torch.cat((isRelated,isRoumer, clarity, usefulness, topics), dim=0))
            #loss_rmse = self.criterion_rmse(result_rmse, torch.cat((clarity, usefulness), dim=0))

            # loss_ce.requres_grad = True
            # loss_rmse.requires_grad = True



            total_loss_ce += loss_ce.item()
            # total_loss_rmse += loss_rmse.item()
            test_loss_ce += loss_ce.item()
            # test_loss_rmse += loss_rmse.item()
            # (loss_ce+loss_rmse).backward()
            loss_ce.backward()
            # loss_ce.backward(retain_graph=True)
            # loss_rmse.backward(retain_graph=True)
            if idx % self.hparams.batchsize == 0 and idx > 0: #pesudo minibatch
                self.optimizer.step()
                self.optimizer.zero_grad()
            if idx % 250 == 0:
                self.hparams.logger.info('Epoch {}, Batch {} : Avg CE Loss {}, Avg RMSE Loss {}'.format(self.epoch, idx, test_loss_ce/500, test_loss_rmse/500))
                test_loss_ce = 0
                test_loss_rmse = 0
        self.epoch += 1
        return (total_loss_rmse+total_loss_ce) / len(dataset)

    def eval(self, dataset):
        self.model.eval()
        inComplete = 0
        with torch.no_grad():
            total_loss_ce, total_loss_rmse = 0.0, 0.0
            # indices = torch.randperm(len(dataset), dtype=torch.long, device=self.device)
            test_loss_ce, test_loss_rmse = 0.0, 0.0
            results = []
            for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
                doc, tags = dataset[idx]
                # doc = [sent[:64] for sent in doc]
                # doc = doc[:256]
                isRelated, isRoumer, clarity, usefulness, topics = tags
                doc = pad_sequence(doc).transpose(0, 1)  # padding document
                result = self.model(doc)
                # loss_ce = self.criterion(result_ce, torch.cat((isRelated, topics), dim=0))
                loss_ce = self.criterion(result, torch.cat((isRelated, isRoumer, clarity, usefulness, topics), dim=0))
                # loss_rmse = self.criterion_rmse(result_rmse, torch.cat((clarity, usefulness), dim=0))

                # loss_ce.requres_grad = True
                # loss_rmse.requires_grad = True

                total_loss_ce += loss_ce.item()
                # total_loss_rmse += loss_rmse.item()
                test_loss_ce += loss_ce.item()
                # test_loss_rmse += loss_rmse.item()
                # (loss_ce+loss_rmse).backward()
                # loss_ce.backward()
                # loss_ce.backward(retain_graph=True)
                # loss_rmse.backward(retain_graph=True)
                if idx % self.hparams.batchsize == 0 and idx > 0:  # pesudo minibatch
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if idx % 250 == 0:
                    self.hparams.logger.info(
                        'Epoch {}, Batch {} : Avg CE Loss {}, Avg RMSE Loss {}'.format(self.epoch, idx,
                                                                                       test_loss_ce / 500,
                                                                                       test_loss_rmse / 500))
                    test_loss_ce = 0
                    test_loss_rmse = 0
        return (total_loss_rmse+total_loss_ce) / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        inComplete = 0
        with torch.no_grad():
            total_loss_ce, total_loss_rmse = 0.0, 0.0
            # indices = torch.randperm(len(dataset), dtype=torch.long, device=self.device)
            test_loss_ce, test_loss_rmse = 0.0, 0.0
            results = []
            for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
                doc, tags = dataset[idx]
                # doc = [sent[:64] for sent in doc]
                # doc = doc[:256]
                # for sent in doc:
                #     sent.to(self.hparams.device)
                # for tag in tags:
                #     tag.to(self.hparams.device)
                isRelated,isRoumer, clarity, usefulness, topics = tags
                doc = pad_sequence(doc).transpose(0, 1)  # padding document
                #result_ce, result_rmse = self.model(doc)
                predictions = self.model(doc)

                results.append(predictions)
                # print(torch.cat((isRelated, topics, clarity, usefulness), dim=0))
                loss_ce = self.criterion(predictions, torch.cat((isRelated, isRoumer, clarity, usefulness, topics), dim=0))
                # loss_rmse = self.criterion_rmse(result_rmse, torch.cat((clarity, usefulness), dim=0))

                # loss_ce.requres_grad = True
                # loss_rmse.requires_grad = True

                total_loss_ce += loss_ce.item()
                # total_loss_rmse += loss_rmse.item()
                test_loss_ce += loss_ce.item()
                # test_loss_rmse += loss_rmse.item()
                if idx % 250 == 0:
                    self.hparams.logger.info(
                        'Testing: Epoch {}, Batch {} : Avg CE Loss {}, Avg RMSE Loss {}'.format(self.epoch, idx,
                                                                                       test_loss_ce / 500,
                                                                                       test_loss_rmse / 500))
                    test_loss_ce = 0
                    test_loss_rmse = 0
        return torch.stack(results, dim=0)
