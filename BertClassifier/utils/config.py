
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bert Classifier Project for Covid-19')
    # data arguments
    parser.add_argument('--data', default='data/0421/',
                        help='path to dataset')
    # parser.add_argument('--glove', default='data/glove/',
    #                     help='directory with GLOVE embeddings')
    parser.add_argument('--bert_path', default='dependencies/bert/..',
                        help='directory where bert model resides')
    # parser.add_argument('--expname', type=str, default='rnng-discriminative',
    #                     help='Name to identify experiment')
    # model arguments
    parser.add_argument('--input_dim', default=300, type=int,
                        help='Size of input word vector')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--category_dim', default=300, type=int,
                        help='Size of input syntactic category')
    parser.add_argument('--mem_dim', default=300, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--hidden_dim', default=300, type=int,
                        help='Size of classifier MLP')
    parser.add_argument('--num_classes', default=5, type=int,
                        help='Number of classes in dataset')
    parser.add_argument('--freeze_embed', action='store_true',
                        help='Freeze word embeddings')
    # finetuning arguments
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adam',
                        help='optimizer (default: adagrad)')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()#args=['--data', '/tmp/tree2seq_data/data/tree2seq/'])
    return args
