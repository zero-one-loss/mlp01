import argparse

parser = argparse.ArgumentParser(description='SCD01 Binary-classes')

# string
parser.add_argument('--target', default='scd.pkl', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--type', default='scd', type=str,
                    help='scd or svm')
parser.add_argument('--source', default='svm.pkl', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--gpu', default='1', type=str,
                    help='gpu device')
parser.add_argument('--metrics', default='balanced', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--mc', default='mean', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--ittype', default='one', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--comment', default='nothing', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--dataset', default='mnist', type=str,
                    help='dataset')
parser.add_argument('--version', default='v1', type=str,
                    help='scd version')
parser.add_argument('--init', default='normal', type=str,
                    help='scd version')

# int
parser.add_argument('--num-iters', default=100, type=int,
                    help='number of iters in each vote training')
parser.add_argument('--updated-features', default=128, type=int,
                    help='number of features will be update in each iteration')
parser.add_argument('--round', default=1, type=int,
                    help='number of vote')
parser.add_argument('--interval', default=10, type=int,
                    help='number of neighbours will be considered '
                         'in bias choosing')
parser.add_argument('--n-jobs', default=2, type=int,
                    help='number of processes')
parser.add_argument('--num-gpus', default=1, type=int,
                    help='number of GPUs')
parser.add_argument('--hidden-nodes', default=5, type=int,
                    help='number of processes')
parser.add_argument('--iters', default=200, type=int,
                    help='ratio of rows in each iteration')
parser.add_argument('--epoch', default=20, type=int,
                    help='training epoch')
parser.add_argument('--aug-epoch', default=20, type=int,
                    help='attack epoch')
parser.add_argument('--train-size', default=200, type=int,
                    help='sample size')
parser.add_argument('--random-sign', default=0, type=int,
                    help='change lambda\'s sign')
parser.add_argument('--width', default=500, type=int,
                    help='number of iters in each vote training')
parser.add_argument('--updated-nodes', default=1, type=int,
                    help='change lambda\'s sign')
parser.add_argument('--h-times', default=1, type=int,
                    help='number of iters in each vote training')
parser.add_argument('--localit', default=100, type=int,
                    help='number of iters in each vote training')
parser.add_argument('--seed', default=2018, type=int,
                    help='random seed')
parser.add_argument('--oracle-size', default=256, type=int,
                    help='random seed')
parser.add_argument('--n_classes', default=2, type=int,
                    help='number of classes')

# float
parser.add_argument('--nrows', default=0.75, type=float,
                    help='ratio of rows in each iteration')
parser.add_argument('--alpha', default=0, type=float,
                    help='ratio of rows in each iteration')
parser.add_argument('--nfeatures', default=1, type=float,
                    help='ratio of features in each vote')
parser.add_argument('--w-inc', default=0.17, type=float,
                    help='weights increments')
parser.add_argument('--w-inc1', default=0.17, type=float,
                    help='weights increments')
parser.add_argument('--w-inc2', default=0.02, type=float,
                    help='weights increments')
parser.add_argument('--eps', default=1, type=float,
                    help='epsilon in adversarial training')
parser.add_argument('--epsilon', default=1, type=float,
                    help='epsilon')
parser.add_argument('--Lambda', default=0.1, type=float,
                    help='ratio of features in each vote')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate for substitute model')
parser.add_argument('--c', default=0.01, type=float,
                    help='c')
parser.add_argument('--b-ratio', default=0.2, type=float,
                    help='ratio of iterations of updating by balanced metrics')

# bool
parser.add_argument('--adv-train', action='store_true',
                    help='Run adversarail training')
parser.add_argument('--no-eval', action='store_true',
                    help='evaluation')
parser.add_argument('--verbose', action='store_true',
                    help='show intermediate acc output')
parser.add_argument('--dual', action='store_true', help='Dual')
parser.add_argument('--save', action='store_true', help='Dual')
parser.add_argument('--deep-search', action='store_true', help='Dual')
parser.add_argument('--alter-metrics', action='store_true', help='Dual')

args = parser.parse_args()