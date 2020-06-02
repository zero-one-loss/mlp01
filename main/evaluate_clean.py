import numpy as np
import pickle
import sys
sys.path.append('..')
from tools import args, load_data
import time
import os
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import pandas as pd


if __name__ == '__main__':
    np.random.seed(args.seed)

    # print('training data size: ')
    # print(train.shape)
    # print('testing data size: ')
    # print(test.shape)

    save_path = 'checkpoints'

    model_list = [
        # 'scd01_single_vote',
        # 'scd01_32_vote',
        # 'scd01_v12_adv',
        # 'scd01_v13_adv',
        # 'scd01_v16_adv',
        # 'scd01_v17_adv',
        # 'scd01mlp_single_vote',
        # 'scd01mlp_32_vote_bs',
        # 'scd01_32_vote_ub_v9_02_100',
        # 'scd01mlp_32_vote_ub_v10_075_1000',
        'mlp1vall_32vote',
        'mlp1vall'
        # 'scd01_32_vote_bp_v9_075_10',
        # 'scd01mlp_v12_adv',
        # 'scd01mlp_v13_adv',
        # 'scd01mlp_v16_adv',
        # 'scd01mlp_v17_adv',
        # 'svm',
        # 'mlp',
        # 'scd01mlp_32_vote_40',
        # 'scd01mlp_32_vote_60',
        # 'scd01mlp_32_vote_80',
        # 'scd01mlp_32_vote_100',
        # 'mlp_40',
        # 'mlp_60',
        # 'mlp_80',
        # 'mlp_100',
    ]

    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

    # if args.dataset == 'cifar10':
    for model in model_list:
        print(model)
        with open(os.path.join(save_path, '%s_' % args.dataset +model+'.pkl'), 'rb') as f:
            scd = pickle.load(f)
        yp = scd.predict(test)
        print('test  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
        yp = scd.predict(train)
        print('train  Accuracy: ', accuracy_score(y_true=train_label, y_pred=yp))

            # yp_ = scd.predict(test, distance=True, all=True)
            # print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp_.argmax(1)))
            # yp = scd.predict(test)
            # print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))

            # for i in range(10):
            #     print('class %d: ' % i)
            #     y_train = np.zeros_like(train_label)
            #     y_train[train_label == i] = 1
            #     y_test = np.zeros_like(test_label)
            #     y_test[test_label == i] = 1
            #     yp0 = scd.scds['%d' % i].predict(train, all=True)
            #
            #     for j in range(32):
            #         acc = balanced_accuracy_score(y_true=y_train, y_pred=yp0[:, j])
            #         print('vote %d acc: %.5f' % (j, acc))
            #         if acc < 0.9:
            #             print('vote %d acc is below' % j)

