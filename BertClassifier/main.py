from utils import DRNNGDataset
import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from config import parse_args
from utils import globalVar
from utils import utils
from utils import Vocab, DetVocab
from utils import Constant
from model import rnng
from helpers import Trainer
import pydevd_pycharm


def main():
    global args
    args = parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
#     logger = globalVar.logger
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

    Hpa

    globalVar.logger = logger

    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()

    #
    device = torch.device("cuda:0" if args.cuda else "cpu")
#     device = torch.device("cpu")
    globalVar.device = device

    print(globalVar.device)

    if args.debug:
        pydevd_pycharm.settrace('localhost', port=12025, stdoutToServer=True, stderrToServer=True)


    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)


    #vocab buildup
    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    rnng_vocab_file = os.path.join(args.data, 'rnng.vocab')
    if not os.path.isfile(rnng_vocab_file):
        token_files= [os.path.join(split, 'tokens') for split in [train_dir]]  #Using unknowified tokens
        rnng_vocab_file = os.path.join(args.data, 'rnng.vocab')
        utils.build_vocab(token_files, rnng_vocab_file)
    vocab = Vocab(filename=rnng_vocab_file, data=[Constant.UNK_WORD, Constant.EOS_WORD]) # NO special token is needed as unknownification has been done in preprocessing stage
    globalVar.vocab = vocab

    act_vocab_file = os.path.join(args.data, 'act.vocab')
    if not os.path.isfile(act_vocab_file):
        token_files= [os.path.join(split, 'actions') for split in [train_dir]]
        rnng_vocab_file = os.path.join(args.data, 'act.vocab')
        utils.build_vocab(token_files, act_vocab_file)
    act_vocab = Vocab(filename=act_vocab_file, data=[Constant.PAD_WORD]) # NO special token is needed as unknownification has been done in preprocessing stage
    globalVar.act_vocab = act_vocab

    pos_vocab_file = os.path.join(args.data, 'pos.vocab')
    if not os.path.isfile(pos_vocab_file):
        token_files= [os.path.join(split, 'POSs') for split in [train_dir]]
        pos_vocab_file = os.path.join(args.data, 'pos.vocab')
        utils.build_vocab(token_files, pos_vocab_file)
    pos_vocab = DetVocab(filename=pos_vocab_file) # NO special token is needed as unknownification has been done in preprocessing stage
    globalVar.pos_vocab = pos_vocab

    nts_vocab_file = os.path.join(args.data, 'nts.vocab')
    if not os.path.isfile(nts_vocab_file):
        NTs = set(filter(lambda x: x.startswith('NT'), act_vocab.idxToLabel.values()))
        nts_vocab_file = os.path.join(args.data, 'nts.vocab')
        utils.build_vocab_set(NTs, nts_vocab_file)
    nts_vocab = DetVocab(filename=nts_vocab_file) # NO special token is needed as unknownification has been done in preprocessing stage
    globalVar.nts_vocab = nts_vocab


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
    model_drnng = rnng.DRNNG_LSTM(input_act=300, input_buf=300, component_size=300, hid_act=300, hid_buf=300, hid_stack=300)
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
