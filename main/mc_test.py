import numpy as np
import pickle
import sys

sys.path.append('..')
from tools import args, load_data
import time
import os
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

    if args.dataset == 'mnist':
        model_list = [
            'scd01_32_br02_nr075_ni10',
            'scd01mlp_32_br02_nr075_ni500_i1',
            'scd01mlp_32_br05_nr075_ni1000',
            'svm',
            'mlp',

        ]
    elif args.dataset == 'cifar10':
        model_list = [
            'scd01_32_br02_nr075_ni1000',
            'scd01mlp_32_br05_nr075_ni500_i1',
            'scd01mlp_32_br05_nr075_ni1000',
            'svm',
            'mlp',

        ]
    elif args.dataset == 'imagenet':
        model_list = [
            'scd01_32_br05_nr075_ni500',
            'scd01mlp_32_br05_nr075_ni500_i1',
            'scd01mlp_32_br02_nr075_ni1000',
            'svm_0001',
            'mlp400_001',

        ]

    dh = []
    for model in model_list:
        print('Model: ', model)
        with open('checkpoints/%s_%s.pkl' % (args.dataset, model), 'rb') as f:
            scd = pickle.load(f)
        # scd.bs(train, train_label)

        train_acc = accuracy_score(y_true=train_label, y_pred=scd.predict(train))
        test_acc = accuracy_score(y_true=test_label, y_pred=scd.predict(test))
        print('train acc: ', train_acc)
        print('test acc: ', test_acc)
        # train_acc = accuracy_score(y_true=train_label, y_pred=scd.predict(train, distance=True))
        # test_acc = accuracy_score(y_true=test_label, y_pred=scd.predict(test, distance=True))
        # print('train acc: ', train_acc)
        # print('test acc: ', test_acc)
        # with open('checkpoints/%s_%s.pkl' % (args.dataset, model), 'wb') as f:
        #     pickle.dump(scd, f)
        #
        # yp = scd.decision_function(test)
        # yp = (np.sign(yp) + 1) // 2
        # for i in range(10):
        #     y_test = np.zeros_like(test_label)
        #     y_test[test_label == i] = 1
        #     print('acc: ', (yp[:, i] == y_test).mean())
        #     print(('balanced acc: ', balanced_accuracy_score(y_test, yp[:, i])))
        # t = yp[np.arange(yp.shape[0]), test_label]
        # t = (np.sign(t) + 1) // 2
        # print(t.sum())
        # train_prob = scd.predict(train, kind='vote', all=True).reshape((train.shape[0], -1))
        # test_prob = scd.predict(test, kind='vote', all=True).reshape((test.shape[0], -1))

        # for hidden in [10, 20, 30]:
        #     for act in ['logistic', 'relu']:
        #         for lr in [0.01, 0.001]:
        #             print(hidden, act)
        #             mlp = MLPClassifier(hidden_layer_sizes=(hidden,), activation=act, solver='sgd', alpha=0.0001,
        #                             batch_size='auto',
        #                             learning_rate='constant', learning_rate_init=lr, power_t=0.5, max_iter=1000,
        #                             shuffle=True,
        #                             random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
        #                             nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
        #                             beta_2=0.999,
        #                             epsilon=1e-08, n_iter_no_change=10)
        #
        #             mlp.fit(train_prob, train_label)
        #
        #
        #             train_acc = accuracy_score(y_true=train_label, y_pred=mlp.predict(train_prob))
        #             test_acc = accuracy_score(y_true=test_label, y_pred=mlp.predict(test_prob))
        #             print('train acc: ', train_acc)
        #             print('test acc: ', test_acc)
        # clf = KNeighborsClassifier(5, weights='distance')
        # clf.fit(train_prob, train_label)
        # p_d, p_i = clf.kneighbors(test_prob, 5)