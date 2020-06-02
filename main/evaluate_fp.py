import numpy as np
import pickle
import sys
sys.path.append('..')
from tools import args, load_data
import time
import os
from sklearn.metrics import accuracy_score
import pandas as pd






def flip_prob(predictions, noise_perturbation=False):

    prev_pred = predictions[0]
    result_for_vid = []

    for index in range(1, predictions.shape[0]):
        result_for_vid.append((predictions[index] != prev_pred).sum())
        if not noise_perturbation:
            prev_pred = predictions[index]

    return np.sum(result_for_vid) / ((predictions.shape[0] - 1) * predictions.shape[1])


    return result


if __name__ == '__main__':



    save_path = 'checkpoints'

    if args.dataset == 'mnist':
        model_list = [
            'scd01_32_br02_nr075_ni10',
            'scd01_32_br02_nr075_ni500_i1',
            'scd01mlp_32_br02_nr075_ni500_i1',
            # 'scd01mlp_32_br05_nr075_ni500_i1',
            'scd01mlp_32_br02_nr075_ni1000',
            'scd01mlp_32_br02_nr075_ni1000_i1',
            # 'scd01mlp_32_br05_nr075_ni1000',
            'svm',
            'mlp',
            'mlp1vall'

        ]
    elif args.dataset == 'cifar10':
        model_list = [
            # 'scd01_32_br02_nr075_ni1000',
            # 'scd01mlp_32_br02_nr075_ni500_i1',
            # 'scd01mlp_32_br05_nr075_ni500_i1',
            # 'scd01mlp_32_br02_nr075_ni1000',
            # 'scd01mlp_32_br05_nr075_ni1000',
            # 'svm',
            # 'mlp',
        # 'scd01_32_br02_nr075_ni500_i1',
        # 'scd01mlp_32_br02_nr075_ni1000_i1',
        # 'mlp1vall'

        ]
    elif args.dataset == 'imagenet':
        model_list = [
            'scd01_32_br05_nr075_ni500',
            'scd01mlp_32_br02_nr075_ni500_i1',
            'scd01mlp_32_br05_nr075_ni500_i1',
            'scd01mlp_32_br02_nr075_ni1000',
            'scd01mlp_32_br05_nr075_ni1000',
            'svm_0001',
            'mlp400_001',

        ]
    # Corruption

    # train, test, train_label, test_label = load_data(args.dataset)

    if args.dataset == 'cifar10':
        curdir = os.getcwd()
        os.chdir('../../scd01mc/data/cifar10/')
        labels = np.load('test_label.npy')
        os.chdir(curdir)
        os.chdir('../data/CIFAR-10-P')
        files = [i for i in os.listdir() if 'labels' not in i]
        datas = [np.load(i) for i in files]
        os.chdir(curdir)

        logs = {}
        logs['type'] = [file[:-4] for file in files]

        for model in model_list:
            # print('model: ', model)
            fp = []
            with open(os.path.join(save_path, '%s_' % args.dataset + model + '.pkl'), 'rb') as f:
                scd = pickle.load(f)
            for i, data in enumerate(datas):
                # print('type: ', files[i][:-4])
                temp_labels = np.stack([labels for i in range(data.shape[1])], axis=1).reshape((-1, ))
                batch_data = data.reshape((-1, 32, 32, 3))
                # index = np.nonzero(temp_labels < 2)[0]
                test_data = batch_data.astype(np.float32).reshape((-1, 32 * 32 * 3)) / 255
                test_label = temp_labels
                prediction = scd.predict(test_data)
                prediction = prediction.reshape((-1, data.shape[1]))
                print('difficulty: ', data.shape[1])
                current_fp = flip_prob(prediction.T, True if 'noise' in files[i] else False)
                fp.append(current_fp)
                print('%s %s flip probability: ' % (model, files[i][:-4]), current_fp)
            logs[model] = fp

            pd.DataFrame(logs).to_csv('perturbation_fp_%s.csv' % args.dataset, index=False)


    elif args.dataset == 'imagenet':
        curdir = os.getcwd()
        os.chdir('../data/imagenet-p')
        files = os.listdir()
        os.chdir(curdir)

        logs = {}
        logs['type'] = files

        for model in model_list:
            print(model)
            acc = []
            with open(os.path.join(save_path, '%s_' % args.dataset + model + '.pkl'), 'rb') as f:
                scd = pickle.load(f)

            os.chdir('../data/imagenet-P')
            fp = []
            for file in files:
                print(file)
                data = np.load('%s.npy' % file)
                labels = np.load('%s_label.npy' % file)
                labels = np.stack([labels for i in range(data.shape[1])], axis=1).reshape((-1,))
                test_data = data.reshape((-1, 3, 224, 224))
                test_data = test_data.reshape((-1, 224 * 224 * 3))
                prediction = scd.predict(test_data)
                prediction = prediction.reshape((-1, data.shape[1]))
                current_fp = flip_prob(prediction.T, True if 'noise' in file else False)
                fp.append(current_fp)
                print('%s %s flip probability: ' % (model, file), current_fp)
            logs[model] = fp
            os.chdir(curdir)

        pd.DataFrame(logs).to_csv('perturbation_fp_%s.csv' % args.dataset, index=False)