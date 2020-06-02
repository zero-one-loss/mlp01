import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
sys.path.append('..')
from tools import args, save_checkpoint, print_title, load_data
from sklearn.metrics import accuracy_score, balanced_accuracy_score



class MLP_pytorch(nn.Module):
    def __init__(self, in_node=28*28, num_classes=10):
        super(MLP_pytorch, self).__init__()
        # self.bn0 = nn.InstanceNorm1d(in_node)
        self.linear1 = nn.Linear(in_node, 20, bias=True)
        # self.bn1 = nn.InstanceNorm1d(200)
        # self.linear2 = nn.Linear(200, 200, bias=True)
        self.linear3 = nn.Linear(20, num_classes, bias=True)
        # self.apply(_weights_init)

    def forward(self, x, features=False):

        out = torch.sigmoid(self.linear1(x))
        out = torch.sigmoid(self.linear3(out))
        return out


def get_mlp_adv(model, dataset, num_classes, test, test_label, epsilon):
    with open('checkpoints/%s_%s.pkl' % (dataset, model), 'rb') as f:
        mlp_sklearn = pickle.load(f)

    mlp = MLP_pytorch(in_node=mlp_sklearn.coefs_[0].shape[0], num_classes=mlp_sklearn.intercepts_[1].shape[0])
    mlp.linear1.weight = nn.Parameter(torch.Tensor(mlp_sklearn.coefs_[0].T))
    mlp.linear3.weight = nn.Parameter(torch.Tensor(mlp_sklearn.coefs_[1].T))
    mlp.linear1.bias = nn.Parameter(torch.Tensor(mlp_sklearn.intercepts_[0].T))
    mlp.linear3.bias = nn.Parameter(torch.Tensor(mlp_sklearn.intercepts_[1].T))



    data = torch.Tensor(test).requires_grad_()
    labels = torch.LongTensor(test_label)
    criterion = nn.CrossEntropyLoss()

    yp = mlp(data)
    yp = torch.cat([1-yp, yp], dim=1)
    # print(yp.shape)
    loss = criterion(yp, labels)
    loss.backward()
    grads = data.grad
    # yp = mlp_sklearn.predict(test)
    adv = (data + epsilon * torch.sign(grads)).clamp_(0, 1)

    return adv.data.numpy()

def get_scd_adv(model, dataset, test, test_label, epsilon):
    with open('checkpoints/%s_%s.pkl' % (dataset, model), 'rb') as f:
        scd = pickle.load(f)

    test_data_adv = np.zeros_like(test)
    # yp = scd.predict(test, kind='best')
    test_data_adv[:, scd.best_w_index] -= epsilon * np.sign(scd.best_w.cpu().numpy().reshape((1, -1))) \
                                          * (test_label.reshape((-1, 1)) * 2 - 1)
    test_data_adv += test
    np.clip(test_data_adv, 0, 1, out=test_data_adv)

    return test_data_adv


def scd01mlp_projection():
    yp1 = (torch.sign(x[:, scd.best_w_index].matmul(scd.best_w1) +
                      scd.best_b1.view((1, -1))) + 1) / 2
    yp = (torch.sign(yp1.matmul(scd.best_w2) +
                     scd.best_b2) + 1) // 2

def get_scd01mlp_adv(model, dataset, test, test_label, epsilon):
    with open('checkpoints/%s_%s.pkl' % (dataset, model), 'rb') as f:
        scd = pickle.load(f)

    index = []
    test_data_adv = np.zeros(shape=(test.shape[0], test.shape[1], scd.hidden_nodes))
    hidden_projection = (np.sign(test[:, scd.best_w_index].dot(scd.best_w1.cpu().numpy()) +
                                    scd.best_b1.cpu().numpy().reshape((1, -1))) + 1) / 2
    # yp = scd.predict(test, kind='best')
    yp = []
    for j in range(scd.hidden_nodes):
        # print(j)
        test_data_adv[:, :, j] -= epsilon * np.sign(scd.best_w1[:, j].cpu().numpy().reshape((1, -1))) \
                                  * (hidden_projection[:, j].reshape((-1, 1)) * 2 - 1)
        test_data_adv[:, :, j] += test
        np.clip(test_data_adv, 0, 1, out=test_data_adv)
        yp.append(scd.predict(test_data_adv[:, :, j], kind='best'))
    yp = np.stack(yp, axis=1)
    # print(yp.shape)
    bool_matrix = (yp != np.stack([test_label] * scd.hidden_nodes, axis=1)).astype(np.int8)
    # print(bool_matrix.shape)

    for i in range(test.shape[0]):
        temp_bool = np.nonzero(bool_matrix[i])[0]
        if temp_bool.size != 0:
            index.append(np.random.choice(temp_bool))
        else:
            index.append(0)
    # print(len(index))
    adv = test_data_adv[np.arange(test.shape[0]), :, index]
    # print(index)
    # print(np.nonzero(index)[0].shape)
    # print(adv.shape)
    return adv



def get_svm_adv(model, dataset, test, test_label, epsilon):
    with open('checkpoints/%s_%s.pkl' % (dataset, model), 'rb') as f:
        scd = pickle.load(f)
        
    test_data_adv = np.zeros_like(test)
    test_data_adv -= epsilon * (test_label.reshape((-1, 1)) * 2 - 1) * np.sign(scd.coef_)
    test_data_adv += test
    np.clip(test_data_adv, 0, 1, out=test_data_adv)

    return test_data_adv

if __name__ == '__main__':

    if args.dataset == 'mnist':
        model_list = [
            'scd01mlp_32_br02_nr075_ni1000_i1',
            'scd01_32_br02_nr075_ni1000_i1',
            'svm',
            'mlp',
            'mlp1vall_32vote',
            'svm1vall_32vote',
        ]

    elif args.dataset == 'cifar10':
        model_list = [
            'scd01_32_br02_nr075_ni1000_i1',
            'scd01mlp_32_br02_nr075_ni1000_i1',
            'svm',
            'mlp',
            'mlp1vall_32vote',
            'svm1vall_32vote',
        ]

    elif args.dataset == 'imagenet':
        model_list = [
            'scd01_32_br02_nr075_ni1000_i1',
            'scd01mlp_32_br02_nr075_ni1000_i1',
            'svm',
            'mlp'
        ]


    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)



    for source_model in model_list:
        try:
            if source_model == 'mlp':
                test_adv = get_mlp_adv(source_model, args.dataset, args.n_classes, test, test_label, args.epsilon)
            elif 'scd01_' in source_model:
                test_adv = get_scd_adv(source_model, args.dataset, test, test_label, args.epsilon)
            elif 'svm' in source_model:
                test_adv = get_svm_adv(source_model, args.dataset, test, test_label, args.epsilon)
            elif 'scd01mlp' in source_model:
                test_adv = get_scd01mlp_adv(source_model, args.dataset, test, test_label, args.epsilon)
            for target_model in model_list:

                with open('checkpoints/%s_%s.pkl' % (args.dataset, target_model), 'rb') as f:
                    model = pickle.load(f)
                if 'scd' in target_model:
                    yp = model.predict(test, kind='best')
                    print('Target model (%s) clean scc: %.5f' % (target_model,
                                                                 accuracy_score(y_true=test_label, y_pred=yp)))

                    yp_adv = model.predict(test_adv, kind='best')
                    print('Source model (%s) Target model (%s) adv scc: %.5f' % (source_model, target_model,
                                                                 accuracy_score(y_true=test_label, y_pred=yp_adv)))
                    yp = model.predict(test, kind='vote')
                    print('Target model (%s) clean scc: %.5f' % (target_model,
                                                                 accuracy_score(y_true=test_label, y_pred=yp)))

                    yp_adv = model.predict(test_adv, kind='vote')
                    print('VOTE Source model (%s) Target model (%s) adv scc: %.5f' % (source_model, target_model,
                                                                 accuracy_score(y_true=test_label, y_pred=yp_adv)))
                else:
                    yp = model.predict(test)
                    print('Target model (%s) clean scc: %.5f' % (target_model,
                                                                 accuracy_score(y_true=test_label, y_pred=yp)))

                    yp_adv = model.predict(test_adv)
                    print('Source model (%s) Target model (%s) adv scc: %.5f' % (source_model, target_model,
                                                                                 accuracy_score(y_true=test_label,
                                                                                                y_pred=yp_adv)))
        except:
            continue