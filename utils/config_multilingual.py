
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bert Classifier Project for Covid-19')
    # data arguments
    parser.add_argument('--data', default='data/MultiLingual/0804/',
                        help='path to dataset')
    # parser.add_argument('--glove', default='data/glove/',
    #                     help='directory with GLOVE embeddings')
    parser.add_argument('--bert_path', default='dependencies/bert/L-6_H-768_A-6_E-40_BPE',
                        help='directory where bert model resides')
    parser.add_argument('--train_bert', dest='train_bert', action='store_true'
                        , help="whether to train bert model")
    parser.set_defaults(train_bert=False )
    parser.add_argument('--use_f1', dest='use_f1', action='store_true'
                        , help="whether to train bert model")
    parser.set_defaults(use_f1=False )
    parser.add_argument('--combine_xr', dest='combine_xr', action='store_true'
                        , help="combine with expected recall")
    parser.set_defaults(use_f1=False)
    parser.add_argument('--use_ce', dest='use_ce', action='store_true'
                        , help="whether to train bert model")
    parser.set_defaults(use_ce=False )
    parser.add_argument('--expname', type=str, default='BertClassifier',
                        help='Name to identify experiment')
    parser.add_argument('--expdir', type=str, default='/home/christopher/experiments/covid/',
                        help='Name to identify experiment')
    parser.add_argument('--save', type=str, default='checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--class_config', type=str, default='configs/classes',
                        help='path to checkpoints')
    parser.add_argument('--bertrnn', dest='bertrnn', action='store_true'
                        , help="Use RNN as mixer")
    parser.add_argument('--train_by_chunk', dest='train_by_chunk', action='store_true'
                        , help="Whether to use chunking")
    parser.add_argument('--article_level', dest='article_level', action='store_true'
                        , help="Whether to train it as a complete article")
    parser.add_argument('--article_len', type=int, default=512,
                        help='Maximum article length on AT mode')
    parser.add_argument('--chunk_len_limit', type=int, default=256,
                        help='Maximum article length on AT mode')
    parser.add_argument('--context_size', type=int, default=30,
                        help='Maximum article length on AT mode')
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
    parser.add_argument('--finetuning', action='store_true',
                        help='Finetuning the bert?')
    parser.add_argument('--batchsize', type=int, default=25,
                        help='batchsize')
    parser.add_argument('--decoder_sharing', dest='decoder_sharing', action='store_true'
                        , help="share decoder ?")
    # finetuning arguments
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--sample_count', default=3, type=int,
                        help='Size of input word vector')
    parser.add_argument('--sample_size', default=48, type=int,
                        help='Size of input word vector')
    parser.add_argument('--error_report', default=1000, type=int,
                        help='Size of input word vector')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adam',
                        help='optimizer (default: adagrad)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()#args=['--data', '/tmp/tree2seq_data/data/tree2seq/'])
    return args
