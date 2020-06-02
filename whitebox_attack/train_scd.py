import numpy as np
import pickle
import sys
sys.path.append('..')
from tools import args, save_checkpoint, print_title, load_data

if args.version == 'linear':
    from core.scd01_binary import SCD
    # basic scd 01 loss multi-class one vs all linear classifier

elif args.version == 'mlp':
    from core.scd01mlp_binary import SCD
    # scd mlp 01 multi-class one vs all


import time
from sklearn.metrics import accuracy_score
import torch

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Set Random Seed


    # print information
    et, vc = print_title()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    save = True if args.save else False

    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

    print('training data size: ')
    print(train.shape)
    print('testing data size: ')
    print(test.shape)

    scd_params = {
        'nrows': args.nrows,
        'nfeatures': args.nfeatures,
        'w_inc': args.w_inc,
        'tol': 0.00000,
        'local_iter': args.iters,
        'num_iters': args.num_iters,
        'interval': args.interval,
        'round': args.round,
        'w_inc1': args.w_inc1,
        'updated_features': args.updated_features,
        'n_jobs': args.n_jobs,
        'num_gpus': args.num_gpus,
        'adv_train': True if args.adv_train else False,
        'eps': args.eps,
        'w_inc2': args.w_inc2,
        'hidden_nodes': args.hidden_nodes,
        'evaluation': False if args.no_eval else True,
        'verbose': True if args.verbose else False,
        'width': args.width,
        'metrics': args.metrics,
        'init': args.init,
        'b_ratio': args.b_ratio,
        # 'n_classes': args.n_classes
    }


    scd = SCD(**scd_params)

    a = time.time()
    scd.train(train, train_label, test, test_label)

    print('Cost: %.3f seconds'%(time.time() - a))

    print('Best Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train, kind='best')))
    print('Vote Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train, kind='vote')))
    print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=scd.predict(test, kind='best')))
    print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=scd.predict(test, kind='vote')))

    if save:
        save_path = 'checkpoints'
        save_checkpoint(scd, save_path, args.target, et, vc)