import numpy as np
import pickle
import sys
sys.path.append('..')
from core.svm_ensemble import SVM
from tools import args, save_checkpoint, print_title, load_data
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Set Random Seed
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # print information
    et, vc = print_title()
    save = True if args.save else False

    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

    # t = args.epoch
    # print(t)
    # train_label[train_label != t] = 10
    # train_label[train_label == t] = 1
    # train_label[train_label == 10] = 0
    # test_label[test_label != t] = 10
    # test_label[test_label == t] = 1
    # test_label[test_label == 10] = 0
    print('training data size: ')
    print(train.shape)
    print('testing data size: ')
    print(test.shape)

    dual = True if args.dual else False
    scd = SVM(n_classes=args.n_classes, round=args.round, C=args.c, dual=dual)
    a = time.time()
    scd.fit(train, train_label)
    print('Cost: %.3f seconds'%(time.time() - a))

    print('Best Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
    print('Balanced Train Accuracy: ', balanced_accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
    print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=scd.predict(test)))
    print('Balanced Accuracy: ', balanced_accuracy_score(y_true=test_label, y_pred=scd.predict(test)))

    if save:
        save_path = 'checkpoints'
        save_checkpoint(scd, save_path, args.target, et, vc)