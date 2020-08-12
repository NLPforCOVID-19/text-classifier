import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

# from helpers import Trainer
#
from utils import HParams, DetVocab, parse_args, Trainer, print_evals, get_tags_from_dataset

from model import BertMinMax, BertRNN, BCDataset



def main():

    # hyperparameter holder
    hparams = HParams()
    args = parse_args()
    hparams.set_from_args(args)

    # print(hparams.batchsize)
    # setting up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    hparams.logger = logger

    # setting up device
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    hparams.device = device

    # mode setting
    if args.debug:
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=12025, stdoutToServer=True, stderrToServer=True)
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    # print out arguments
    hparams.logger.debug(args)
    # seeding
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # class vocab buildup
    classes_vocab = DetVocab(filename=args.class_config)
    hparams.classes_vocab = classes_vocab

    #Initialize BERT model (as it includes Jumanpp analyzer)

    # jumanpp = JumanAnalyzer()
    if args.bertrnn:
        model = BertRNN(args.bert_path, classes_vocab.size(), hparams,
                               langs=['de', 'ja', 'fa', 'it', 'pt', 'es', 'fr', 'en', 'vi', 'zh-cn', 'ru', 'ar', 'ko'])
    else:
        model = BertMinMax(args.bert_path, classes_vocab.size(), hparams,
                           langs = ['de', 'ja', 'fa', 'it', 'pt', 'es', 'fr', 'en', 'vi', 'zh-cn', 'ru', 'ar', 'ko'])
    if not args.finetuning:
        model.setoff_bert()
    # print(model.tokenizer.cls_token_id)

    # dataset loading
    train_json = os.path.join(args.data, 'crowdsourcing.train')
    dev_json = os.path.join(args.data, 'crowdsourcing.dev')
    test_json = os.path.join(args.data, 'crowdsourcing.test')

    train_file = os.path.join(args.data, 'train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = BCDataset.loadData(train_json, model.tokenize, model.token2id, device)
        # print(train_dataset[0])
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data:{}'.format(len(train_dataset)))
    # train_dataset.moveto(hparams.device)

    dev_file = os.path.join(args.data, 'dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = BCDataset.loadData(dev_json, model.tokenize, model.token2id, device)
        torch.save(dev_dataset, dev_file)
    logger.debug('==> Size of dev data:{}'.format(len(dev_dataset)))
    # dev_dataset.moveto(hparams.device)

    test_file = os.path.join(args.data, 'test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = BCDataset.loadData(test_json, model.tokenize, model.token2id, device)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data:{}'.format(len(test_dataset)))
    # print(train_dataset.get_langs())
    # exit()
    criterion = nn.NLLLoss()
    model.to(hparams.device), criterion.to(hparams.device)
    if args.optim == 'adam':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)

    trainer = Trainer(hparams, model, criterion, optimizer)
    model_file = os.path.join(args.save, args.expname)
    # print(model.parameters())
    # exit()


    best = float('inf')
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        logger.info('==> Epoch {}, Train \tLoss: {} '.format(epoch, train_loss))
        if train_loss < best:
            torch.save(model.state_dict(), model_file)
        dev_loss = trainer.eval(dev_dataset)
        logger.info('==> Epoch {}, Dev \tLoss: {} '.format(epoch, dev_loss))
        results = trainer.test(test_dataset)
        print_evals(results, get_tags_from_dataset(test_dataset), logger)
        trainer.epoch+=1
        # train_metrics = trainer.test(train_dataset)



if __name__ == "__main__":
    main()
