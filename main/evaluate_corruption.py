import numpy as np
import pickle
import sys
sys.path.append('..')
from tools import args, load_data
import time
import os
from sklearn.metrics import accuracy_score
import pandas as pd






# pick first two classes
np.random.seed(args.seed)

# print('training data size: ')
# print(train.shape)
# print('testing data size: ')
# print(test.shape)


save_path = 'checkpoints'
if args.dataset == 'mnist':
    model_list = [
        # 'scd01_32_br02_nr075_ni10',
        'scd01mlp_32_br02_nr075_ni500_i1',
        # 'scd01mlp_32_br05_nr075_ni1000',
        # 'scd01_32_br02_nr075_ni500_i1',
        # 'scd01mlp_32_br05_nr075_ni500_i1',
        # 'scd01mlp_32_br02_nr075_ni1000',
        'scd01mlp_32_br02_nr075_ni1000_i1',
        # 'scd01mlp80_32_br02_nr075_ni1000_i1'

        'svm',
        'mlp',

        # 'mlp1vall',

    ]
elif args.dataset == 'cifar10':
    model_list = [
        # 'scd01_32_br02_nr075_ni1000',
        'scd01_32_br02_nr075_ni500_i1',
        # 'scd01mlp_32_br02_nr075_ni500_i1',
        # 'scd01mlp_32_br05_nr075_ni500_i1',
        # 'scd01mlp_32_br02_nr075_ni1000',
        # 'scd01mlp_32_br05_nr075_ni1000',
        'scd01mlp_32_br02_nr075_ni1000_i1',
        'svm',
        'mlp',
        'mlp1vall'

    ]
elif args.dataset == 'imagenet':
    model_list = [
        # 'scd01_32_br05_nr075_ni500',
        'scd01mlp_32_br02_nr075_ni500_i1',
        # 'scd01mlp_32_br05_nr075_ni500_i1',
        # 'scd01mlp_32_br02_nr075_ni1000',
        # 'scd01mlp_32_br05_nr075_ni1000',
        # 'svm_0001',
        # 'mlp400_001',

        'scd01mlp_32_br02_nr075_ni1000_i1',
    ]



# Corruption

train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

if args.dataset == 'cifar10':


    curdir = os.getcwd()
    os.chdir('../data/CIFAR-10-C')
    files = [i for i in os.listdir() if 'labels' not in i]
    datas = [np.load(i) for i in files]
    labels = np.load('labels.npy')
    os.chdir(curdir)

    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

    logs = {}
    logs['type'] = [file[:-4] for file in files] + ['clean acc']


    for severity in range(5):
        # index = np.nonzero(labels[severity * 10000: (severity + 1) * 10000] < 2)[0]
        for model in model_list:
            print(model)
            acc = []
            with open(os.path.join(save_path, '%s_' % args.dataset +model+'.pkl'), 'rb') as f:
                scd = pickle.load(f)

            # labels = np.concatenate([scd.predict(test)] * 5, axis=0)

            for i, data in enumerate(datas):
                print(files[i])
                batch_data = data[severity * 10000: (severity + 1) * 10000]
                test_data = batch_data.astype(np.float32).reshape((-1, 32*32*3)) / 255
                yp = scd.predict(test_data)
                print('vote  Accuracy: ', accuracy_score(y_true=labels[severity * 10000: (severity + 1) * 10000], y_pred=yp))
                acc.append(accuracy_score(y_true=labels[severity * 10000: (severity + 1) * 10000], y_pred=yp))

            acc.append(accuracy_score(y_true=test_label, y_pred=scd.predict(test)))
            logs['%s severity lv %d' % (model, severity+1)] = acc


    pd.DataFrame(logs).to_csv('corruptions_%s.csv' % args.dataset, index=False)

    Pertubat


    curdir = os.getcwd()
    os.chdir('../data/cifar10/')
    labels = np.load('test_label.npy')
    os.chdir(curdir)
    os.chdir('../data/CIFAR-10-P')
    files = [i for i in os.listdir() if 'labels' not in i]
    datas = [np.load(i) for i in files]
    os.chdir(curdir)

    logs = {}
    logs['type'] = [file[:-4] for file in files] + ['clean acc']


    for model in model_list:
        acc = []
        with open(os.path.join(save_path, '%s_' % args.dataset + model + '.pkl'), 'rb') as f:
            scd = pickle.load(f)
        for i, data in enumerate(datas):
            print(files[i])
            temp_labels = np.stack([labels for i in range(data.shape[1])], axis=1).reshape((-1, ))
            batch_data = data.reshape((-1, 32, 32, 3))
            # index = np.nonzero(temp_labels < 2)[0]
            test_data = batch_data.astype(np.float32).reshape((-1, 32 * 32 * 3)) / 255

            yp = scd.predict(test_data)
            print('vote  Accuracy: ', accuracy_score(y_true=temp_labels, y_pred=yp))
            acc.append(accuracy_score(y_true=temp_labels, y_pred=yp))
        acc.append(accuracy_score(y_true=test_label, y_pred=scd.predict(test)))
        logs[model] = acc

    pd.DataFrame(logs).to_csv('perturbation_%s.csv'  % args.dataset, index=False)


if args.dataset == 'mnist':
    curdir = os.getcwd()
    os.chdir('../data/mnist_c')
    categories = os.listdir()
    os.chdir(curdir)

    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

    logs = {}
    logs['type'] = [category for category in categories] + ['clean acc']
    for model in model_list:
        print(model)
        acc = []
        with open(os.path.join(save_path, '%s_' % args.dataset + model + '.pkl'), 'rb') as f:
            scd = pickle.load(f)

        for category in categories:
            test_data = np.load('../data/mnist_c/%s/test_images.npy' % category).reshape((-1, 28 * 28)) /255
            # labels = scd.predict(test)
            labels = np.load('../data/mnist_c/%s/test_labels.npy' % category)

            # index = labels < 2
            #
            # test_data = test_data[index]
            # labels = labels[index]

            yp = scd.predict(test_data)
            temp_acc = accuracy_score(y_true=labels, y_pred=yp)
            print('%s: vote  Accuracy: ' % category, temp_acc)
            acc.append(temp_acc)
        clean_acc = accuracy_score(y_true=test_label, y_pred=scd.predict(test))
        acc.append(clean_acc)
        logs[model] = acc

    pd.DataFrame(logs).to_csv('corruption_%s.csv' % args.dataset, index=False)


if args.dataset == 'imagenet':


    # curdir = os.getcwd()
    # os.chdir('../data/imagenet-c')
    # files = [i for i in os.listdir()]
    # os.chdir(curdir)
    #
    # logs = {}
    # logs['type'] = files + ['clean acc']
    #
    # for severity in range(1, 6):
    #
    #     for model in model_list:
    #         print(model)
    #         acc = []
    #         with open(os.path.join(save_path, '%s_' % args.dataset +model+'.pkl'), 'rb') as f:
    #             scd = pickle.load(f)
    #         os.chdir('../data/imagenet-C')
    #         for file in files:
    #             print(file)
    #             test_data = np.load('%s/%s.npy' % (severity, file)).reshape((-1, 224*224*3))
    #             labels = np.load('%s/%s_label.npy' % (severity, file))
    #             yp = scd.predict(test_data)
    #             print('vote  Accuracy: ', accuracy_score(y_true=labels, y_pred=yp))
    #             acc.append(accuracy_score(y_true=labels, y_pred=yp))
    #
    #         acc.append(accuracy_score(y_true=test_label, y_pred=scd.predict(test)))
    #         logs['%s severity lv %d' % (model, severity)] = acc
    #         os.chdir(curdir)
    #
    #
    # pd.DataFrame(logs).to_csv('corruptions_%s.csv' % args.dataset, index=False)
    #


    curdir = os.getcwd()
    os.chdir('../data/imagenet-p')
    files = os.listdir()
    os.chdir(curdir)

    logs = {}
    logs['type'] = files + ['clean acc']


    for model in model_list:
        print(model)
        acc = []
        with open(os.path.join(save_path, '%s_' % args.dataset +model+'.pkl'), 'rb') as f:
            scd = pickle.load(f)

        os.chdir('../data/imagenet-P')

        for file in files:
            print(file)
            test_data = np.load('%s.npy' % file)
            labels = np.load('%s_label.npy' %file)
            labels = np.stack([labels for i in range(test_data.shape[1])], axis=1).reshape((-1, ))
            test_data = test_data.reshape((-1, 3, 224, 224))
            test_data = test_data.reshape((-1, 224*224*3))
            yp = scd.predict(test_data)
            temp_acc = accuracy_score(y_true=labels, y_pred=yp)
            print('%s: vote  Accuracy: ' % file, temp_acc)
            acc.append(temp_acc)
        clean_acc = accuracy_score(y_true=test_label, y_pred=scd.predict(test))
        acc.append(clean_acc)
        logs[model] = acc
        os.chdir(curdir)

    pd.DataFrame(logs).to_csv('perturbation_%s.csv'  % args.dataset, index=False)