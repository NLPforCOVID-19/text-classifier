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

class TrainMode(Enum):
    Train = 0,
    Dev = 1

class Trainer(object):
    def __init__(self, hparams, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.hparams = hparams
        self.model = model
        self.criterion = nn.BCELoss()
        self.criterion_rmse = nn.MSELoss()
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
            # print(dataset[indices[idx]])
            doc, tags = dataset[indices[idx]]
            isRelated, clarity, usefulness, topics = tags
            doc = pad_sequence(doc) # padding document
            # print(doc.size())
            result_ce, result_rmse = self.model(doc)
            # print(result_ce, result_rmse)
            # print(isRelated, topics)
            # print(torch.cat((isRelated, topics), dim=0))
            loss_ce = self.criterion(result_ce, torch.cat((isRelated, topics), dim=0))
            loss_rmse = self.criterion_rmse(result_rmse, torch.cat((clarity, usefulness), dim=0))

            # loss_ce.requres_grad = True
            # loss_rmse.requires_grad = True

            total_loss_ce += loss_ce.item()
            total_loss_rmse += loss_rmse.item()
            test_loss_ce += loss_ce.item()
            test_loss_rmse += loss_rmse.item()
            loss_ce.backward(retain_graph=True)
            loss_rmse.backward(retain_graph=True)
            if idx % self.hparams.batchsize == 0 and idx > 0: #pesudo minibatch
                self.optimizer.step()
                self.optimizer.zero_grad()
            if idx % 500 == 0:
                self.hparams.logger.info('Epoch {}, Batch {} : Avg CE Loss {}, Avg RMSE Loss {}'.format(self.epoch, idx, test_loss_ce/500, test_loss_rmse/500))
                test_loss_ce = 0
                test_loss_rmse = 0
        self.epoch += 1
        return total_loss_rmse+total_loss_ce / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        inComplete = 0
        with torch.no_grad():
            # indices = torch.randperm(len(dataset), dtype=torch.long, device=globalVar.device)
            # test_loss = 0.0
            predictions = []
            for idx in tqdm(range(len(dataset)), desc='Testing epoch ' + str(self.epoch) + ''):
                # print(dataset[indices[idx]])
                status, actions = dataset[idx]
                tokens = deepcopy(status.buffer)
                POSs = deepcopy(status.buffer_pos)
                pred_acts, isComplete = self.decoder.greedy_decode(status)
                # print(pred_acts)
                if isComplete:
                    predictions.append(actSeq2mrg(pred_acts, tokens, POSs))
                else:
                    # predictions.append(actSeq2mrg(pred_acts, tokens, POSs))
                    inComplete+=1
                # prediction = actSeq2mrg(pred_acts, tokens, POSs) if isComplete
                # print(prediction)
                # predictions.append(prediction)

        return predictions, inComplete/len(dataset)


class GTrainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(GTrainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        self.teacher_forcing_ratio = 0.5
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        # CE = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device=globalVar.device)
        test_loss_act = 0.0
        test_loss_word = 0.0
        test_loss_pos = 0.0

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            # print(dataset[indices[idx]])
            status, actions, tokens, POSs = dataset[indices[idx]]

            # hidden_stack, buffer_hidden, action_hidden = self.model.init_hidden()#GRU version
            stack_hidden, buffer_hidden, action_hidden = self.model.init_hidden()  # LSTM version
            status.setHiddens(stack_hidden, buffer_hidden, action_hidden)

            acts_prob = torch.zeros(len(actions), globalVar.act_vocab.size()).to(self.device)
            words_prob = torch.zeros(len(tokens), globalVar.vocab.size()).to(self.device)
            poss_prob = torch.zeros(len(POSs), globalVar.pos_vocab.size()).to(self.device)

            if (self.epoch>-1) or self.epoch>4:#random.random() > self.teacher_forcing_ratio and
                for step in range(len(actions)):
                    act_logits, word_prob, pos_prob = self.model(status)#LSTM
                    # print("on trainer:",word_prob.size(), pos_prob.size())
                    # acts_prob[step] = act_prob
                    # topv, act_topi = act_prob.topk(1)
                    _, action_prob, transition_act = self.model.transit(status, act_logits, word_prob, pos_prob)
                    if transition_act == TransitionAct.GEN:
                        poss_prob[len(status.buffer_pos)-1] = pos_prob
                        words_prob[len(status.buffer) - 1] = word_prob
                    acts_prob[step] = action_prob
            else:  # teacher forcing
                for step in range(len(actions)):
                    act_logits, word_prob, pos_prob = self.model(status)  # LSTM
                    pesudo_acts_prob = torch.zeros(globalVar.act_vocab.size(), device=globalVar.device, dtype=torch.float)
                    pesudo_acts_prob[actions[step]] = 1
                    pesudo_word_prob = torch.zeros(globalVar.vocab.size(), device=globalVar.device, dtype=torch.float)
                    pesudo_word_prob[tokens[step]] = 1
                    pesudo_pos_prob = torch.zeros(globalVar.pos_vocab.size(), device=globalVar.device, dtype=torch.float)
                    pesudo_pos_prob[POSs[step]] = 1
                    _, action_prob, transition_act = self.model.transit(status, pesudo_acts_prob, pesudo_word_prob, pesudo_pos_prob)

                    if transition_act == TransitionAct.GEN:
                        poss_prob[len(status.buffer_pos)-1] = pos_prob
                        words_prob[len(status.buffer) - 1] = word_prob
                    act_scores = torch.log_softmax(act_logits, dim=0)
                    acts_prob[step] = act_scores

            loss_act = self.criterion(acts_prob, actions)
            loss_pos = self.criterion(poss_prob, POSs)
            loss_words = self.criterion(words_prob, tokens)

            loss = sum([loss_act, loss_pos, loss_words])

            loss.requres_grad = True
            total_loss += loss.item()
            test_loss_act += loss_act.item()
            test_loss_pos += loss_pos.item()
            test_loss_word += loss_words.item()
            loss.backward()
            if idx % self.args.batchsize == 0 and idx > 0: #pesudo minibatch
                self.optimizer.step()
                self.optimizer.zero_grad()
            if idx % 500 == 0:
                globalVar.logger.info('Epoch {}, Batch {} : Per-Act Loss {}, Per-Word Loss {}, Per-POS Loss {}'.format(self.epoch, idx, test_loss_act/500, test_loss_word/500, test_loss_pos/500))
                test_loss_act = 0
                test_loss_word = 0
                test_loss_pos = 0
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            indices = torch.randperm(len(dataset), dtype=torch.long, device='cuda')
            preds = list()
            trgs = list()
            for idx in tqdm(range(int(len(dataset))), desc='Testing epoch  ' + str(self.epoch) + ''):
                sent, pst = dataset[indices[idx]]
                sent = sent.to(self.device)
                word_dist = self.model(pst, sent, criterion=nn.NLLLoss())

                #Computing Loss per Act
                loss = self.criterion(word_dist.squeeze(1), sent[1:])
                total_loss += loss.item()

                pred = self.model.beam_decode(pst)[0][0]#first candidate of the batch(of size 1)
                pred, prob = pred[0], pred[1]
                # print("prediction: {}".format(pred))


                # convert back to words
                pred = [globalVar.vocab.convertToLabels([word[0].item()], Constant.EOS) for word in pred]
                pred = ' '.join([word[0] for word in pred][1:])

                preds.append(pred)
                trgs.append(pst)


            # ted = statistics.mean(ted)

            avg_loss = total_loss / len(dataset)

            return {"loss_pa":avg_loss, "ppl_pa": math.exp(avg_loss), "ted": ted}




class BatchTrainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(BatchTrainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        self.teacher_forcing_ratio = 0.5
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        dataset.moveto(globalVar.device)
        padword = torch.tensor(globalVar.act_vocab.convertToId(Constant.PAD_WORD), device=globalVar.device, dtype=torch.long)
        total_loss = 0.0
        # indices = torch.randperm(len(dataset), dtype=torch.long, device=globalVar.device)
        test_loss = 0.0
        batch_sampler = BatchSampler(RandomSampler(dataset), batch_size=25 if self.epoch<6 else 10, drop_last=False)
        batch_cnt = 0
        # for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
        for indices in tqdm(batch_sampler, desc='Training epoch ' + str(self.epoch + 1) + ''):
            # print(dataset[indices[idx]])

            status, actions = dataset[indices]
            batch_size = len(indices)

            actions = pack_sequence(actions, enforce_sorted=False)
            # actions_len = actions[1]



            actions, actions_len = pad_packed_sequence(actions)
            max_len_in_batch = max(actions_len)
            hidden_stack, buffer_hidden, action_hidden, stack_cell, buffer_cell, action_cell = self.model.init_hidden(batch_size)  # LSTM version
            acts_prob = torch.zeros(max_len_in_batch, batch_size, globalVar.act_vocab.size()).to(self.device)
            if (random.random() > self.teacher_forcing_ratio and self.epoch>8) or self.epoch>12 or False:
                for step in range(max_len_in_batch):
                    # print("NTF")
                    # hidden_stack, buffer_hidden, action_hidden, act_prob = self.model(hidden_stack, buffer_hidden, action_hidden, status) #GRU
                    hidden_stack, buffer_hidden, action_hidden, stack_cell, buffer_cell, action_cell, scores = self.model(hidden_stack, buffer_hidden, action_hidden, stack_cell, buffer_cell, action_cell, status)#LSTM
                    for statue, score, idx in zip(status, scores, range(batch_size)): #status_logit_indicator:#zip(status, logits_wrap):
                        # print(logit, idx)
                        _, prob = self.model.transit(statue, score)
                    acts_prob[step] = scores
            else:  # teacher forcing
                for step in range(max_len_in_batch):
                    # print("TF")
                    # hidden_stack, buffer_hidden, action_hidden, act_prob = self.model(hidden_stack, buffer_hidden, action_hidden, status)#GRU
                    hidden_stack, buffer_hidden, action_hidden, stack_cell, buffer_cell, action_cell, logits = self.model(
                        hidden_stack, buffer_hidden, action_hidden, stack_cell, buffer_cell, action_cell,
                        status)  # LSTM

                    pesudo_acts_prob = torch.zeros(batch_size, globalVar.act_vocab.size(), device=globalVar.device,
                                                   dtype=torch.float)

                    transit_samples_idx = [idx for idx in range(batch_size) if actions[step][idx] != padword]
                    for idx in transit_samples_idx:#range(batch_size):
                        # if actions[step][idx] == padword:
                        #     continue
                        pesudo_acts_prob[idx, actions[step][idx]] = 1
                        # print(idx)
                        # print(status[idx])
                        # print(self.model.transit(status[idx], pesudo_acts_prob[idx]))
                        _, action_prob = self.model.transit(status[idx], pesudo_acts_prob[idx])
                    # pesudo_acts_prob = torch.zeros(globalVar.act_vocab.size(), device=globalVar.device, dtype=torch.float)
                    # pesudo_acts_prob[actions[idx][step]] = 1
                    # _, action_prob = self.model.transit(status, pesudo_acts_prob)

                    acts_prob[step] = torch.log_softmax(logits, dim=1)
            # print(acts_prob)
            assert acts_prob.requires_grad==True
            # print(acts_prob.size(), actions.size())
            loss_per_sentence = [self.criterion(acts_prob[:actions_len[idx], idx, :], actions[:actions_len[idx], idx]) for idx in range(batch_size)]
            loss = sum(loss_per_sentence)
            #loss = self.criterion(acts_prob, actions)
            loss.requres_grad = True
            total_loss += loss.item()
            test_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            batch_cnt += 1
            # if idx % self.args.batchsize == 0 and idx > 0: #pesudo minibatch
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()
            if batch_cnt % 20 == 0:
                globalVar.logger.info('Epoch {}, Batch {} : Action Avg Loss {}'.format(self.epoch, batch_cnt, test_loss/(20*batch_size)))
                test_loss = 0
                # batch_cnt = 0
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            indices = torch.randperm(len(dataset), dtype=torch.long, device='cuda')
            preds = list()
            trgs = list()
            for idx in tqdm(range(int(len(dataset))), desc='Testing epoch  ' + str(self.epoch) + ''):
                # print(dataset[indices[idx]])
                sent, pst = dataset[indices[idx]]
                sent = sent.to(self.device)
                word_dist = self.model(pst, sent, criterion=nn.NLLLoss())

                #Computing Loss per Act
                loss = self.criterion(word_dist.squeeze(1), sent[1:])
                total_loss += loss.item()

                pred = self.model.beam_decode(pst)[0][0]#first candidate of the batch(of size 1)
                pred, prob = pred[0], pred[1]
                # print("prediction: {}".format(pred))


                # convert back to words
                pred = [globalVar.vocab.convertToLabels([word[0].item()], Constant.EOS) for word in pred]
                pred = ' '.join([word[0] for word in pred][1:])


                # print("prediction: {}".format(pred))
                # node_pred = ConstituentTreeNode.fromsentence(pred, globalVar.cate_vocab)
                # pred_trg = pred_trg.append((node_pred, pst))
                # print("prediction_str: {}".format(node_pred))
                # print("target_str:{}".format(pst))
                preds.append(pred)
                trgs.append(pst)

                # print("ted: {}".format(ted))
            # preds = map(lambda x: ConstituentTreeNode.fromsentence(x, globalVar.cate_vocab), preds)
            #preds = ConstituentTreeNode.fromsentences(preds, globalVar.cate_vocab)
            # ted = map(lambda x,y: Metrics.ted(x, y), preds, trgs)

            ted = statistics.mean(ted)

            avg_loss = total_loss / len(dataset)

            return {"loss_pa":avg_loss, "ppl_pa": math.exp(avg_loss), "ted": ted}