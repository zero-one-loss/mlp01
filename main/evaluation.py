import numpy as np
import pickle
import sys
sys.path.append('..')
from tools import args, load_data
import time
import os
from sklearn.metrics import accuracy_score
import pandas as pd




if __name__ == '__main__':
    
    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

    if args.dataset == 'mnist':
        model_list = [
            # 'scd01_32_br02_nr075_ni10',
            # 'scd01mlp_32_br02_nr075_ni500_i1',
            # 'scd01mlp_32_br05_nr075_ni500_i1',
            # 'scd01mlp_32_br02_nr075_ni1000',
            # 'scd01mlp_32_br05_nr075_ni1000',
            'scd01_32_br02_nr075_ni500_i1',
            'scd01mlp_32_br02_nr075_ni1000_i1',
            'svm',
            'mlp',
            'mlp1vall',
            'svm1vall_32vote',
            'mlp1vall_32vote',
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
            'scd01_32_br02_nr075_ni500_i1',
            'scd01mlp_32_br02_nr075_ni1000_i1',
            'svm',
            'mlp',
            'mlp1vall',
            'svm1vall_32vote',
            'mlp1vall_32vote',
        ]
    elif args.dataset == 'imagenet':
        model_list = [
            'scd01_32_br02_nr075_ni500_i1',
            'scd01mlp_32_br02_nr075_ni1000_i1',
            # 'scd01mlp_32_br05_nr075_ni500_i1',
            # 'scd01mlp_32_br02_nr075_ni1000',
            # 'scd01mlp_32_br05_nr075_ni1000',
            'scd01_32_br02_nr075_ni500_i1',
            'scd01mlp_32_br02_nr075_ni1000_i1',
            'svm_0001',
            'mlp400_001',
            'mlp1vall_20',

        ]
    dh = []
    for model in model_list:
        print('Model: ', model)
        with open('checkpoints/%s_%s.pkl' % (args.dataset, model), 'rb') as f:
            scd = pickle.load(f)

        train_acc = accuracy_score(y_true=train_label, y_pred=scd.predict(train))
        test_acc = accuracy_score(y_true=test_label, y_pred=scd.predict(test))
        print('train acc: ', train_acc)
        print('test acc: ', test_acc)
        df = pd.read_csv('results/%s/%s_%s_%s_%s_%s.csv' % (args.dataset, args.dataset, model, str(args.epsilon),
                                                            str(args.Lambda), str(args.random_sign)))
        adv_acc = df[[column for column in df.columns if 'adv_acc' in column]].values.flatten()
        substitute_acc = df[[column for column in df.columns if 'substitute' in column]].values.flatten()
        clean_match_rate = df[[column for column in df.columns if 'clean_match_rate' in column]].values.flatten()
        adv_match_rate = df[[column for column in df.columns if 'adv_match_rate' in column]].values.flatten()

        dt = pd.DataFrame(columns=[model, 'train acc: %.4f' % train_acc, 'test acc: %.4f' % test_acc, 'time cost:'])
        dn = pd.DataFrame(columns=[model, 'train acc: %.4f' % train_acc, 'test acc: %.4f' % test_acc, 'time cost:'])
        dn[model] = ['substitute acc']
        dn['train acc: %.4f' % train_acc] = ['adv acc']
        dn['test acc: %.4f' % test_acc] = ['clean match rate']
        dn['time cost:'] = ['adv match rate']
        
        dt[model] = substitute_acc
        dt['train acc: %.4f' % train_acc] = adv_acc
        dt['test acc: %.4f' % test_acc] = clean_match_rate
        dt['time cost:'] = adv_match_rate

        dh.append(pd.concat([dn, dt], axis=0))

    pd.concat(dh, axis=1).to_csv('results/%s_result.csv' % args.dataset, index=False)
    
    