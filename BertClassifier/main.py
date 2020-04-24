import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from helpers import Trainer
import pydevd_pycharm
from .utils import HParams, DetVocab, parse_args


def main():

    # hyperparameter holder
    hparams = HParams()
    args = parse_args()
    hparams.set_from_args(args)

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
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # class vocab buildup
    classes_vocab = DetVocab(filename=args.class_config)
    hparams.classes_vocab = classes_vocab

    #Initialize BERT model (as it includes Jumanpp analyzer)
    model = rnng.DRNNG_LSTM(input_act=300, input_buf=300, component_size=300, hid_act=300, hid_buf=300,
                                  hid_stack=300)
    # dataset loading
    train_file = os.path.join(args.data, 'train')
    dev_file = os.path.join(args.data, 'dev')
    test_file = os.path.join(args.data, 'test')












    #vocab buildup
    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')


    # Load datasets
    train_file = os.path.join(args.data, 'drnng_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = DRNNGDataset.load_data(train_dir, vocab, act_vocab, pos_vocab)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    dev_file = os.path.join(args.data, 'drnng_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = DRNNGDataset.load_data(dev_dir, vocab, act_vocab, pos_vocab)
        torch.save(dev_dataset, dev_file)
    logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))
    test_file = os.path.join(args.data, 'drnng_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = DRNNGDataset.load_data(test_dir, vocab, act_vocab, pos_vocab)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    globalVar.f_compose = rnng.RNNComposingFunc()
    globalVar.f_compose.to(device)

    criterion = nn.NLLLoss()

    emb_file = os.path.join(args.data, 'rnng_emb.pth')
    if False:  # os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constant.UNK_WORD, Constant.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    print(emb.size())
    model_drnng.buf_emb.weight.data.copy_(emb)

    model_drnng.to(globalVar.device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model_drnng.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model_drnng.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model_drnng.parameters()), lr=args.lr, weight_decay=args.wd)
    # metrics = Metrics(args.num_classes)

    # create trainer object for training and testing
    trainer = Trainer(args, model_drnng, criterion, optimizer, device)

    best = float('inf')
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        logger.info('==> Epoch {}, Train \tLoss: {} '.format(epoch, train_loss))
        # train_metrics = trainer.test(train_dataset)



if __name__ == "__main__":
    main()
