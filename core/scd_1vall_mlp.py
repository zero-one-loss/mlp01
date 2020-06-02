import sys

from .scd01mlp_binary import SCD as SMP
import numpy as np
from time import time
from sklearn.neural_network import MLPClassifier

class SCD(object):
    def __init__(self, n_classes=10, smp_params=None):
        self.n_classes = n_classes
        self.n_clf = self.n_classes
        self.scds = {}
        self.smp_params = smp_params
        for i in range(self.n_classes):
            self.scds['%s' % i] = SMP(**smp_params)

    def train(self, train_data, train_label, test_data, test_label):
        for i in range(self.n_classes):
        # for i in [1]:
            print('Start %d (1) vs all' % i)

            y_train = np.zeros_like(train_label)
            y_train[train_label == i] = 1
            y_test = np.zeros_like(test_label)
            y_test[test_label == i] = 1
            self.scds['%s' % i].train(train_data, y_train, test_data, y_test)

    def predict(self, data, kind='vote', all=False, distance=False):
        # if self.mlp is not None:
        #     yp = np.zeros((data.shape[0], self.scds['0'].round, self.n_classes), dtype=np.float64)
        #     for i in range(self.n_classes):
        #         yp[:, :, i] = self.scds['%s' % i].predict(data, kind='vote', all=True)
        #     data_prob = yp.reshape((yp.shape[0], -1))
        #     return self.mlp.predict(data_prob)
        if distance:
            yp = np.zeros((data.shape[0], self.n_classes), dtype=np.float64)

            for i in range(self.n_classes):
                yp[:, i] = self.scds['%s' % i].predict_projection(data, kind='vote', all=True).clip(0,1000).sum(axis=1)

            if all:
                return yp
            return  yp.argmax(axis=1)
        if kind == 'best':
            yp = np.zeros((data.shape[0], self.n_classes), dtype=np.float64)
            for i in range(self.n_classes):
                yp[:, i] = self.scds['%s' % i].predict(data, kind='best')

            if all:
                return yp
            yp = yp.argmax(axis=1)

        elif kind == 'vote':
            if all:
                yp = np.zeros((data.shape[0], self.scds['0'].round, self.n_classes), dtype=np.float64)
                for i in range(self.n_classes):
                    yp[:,  :, i] = self.scds['%s' % i].predict(data, kind='vote', all=True)
                return yp
            yp = np.zeros((data.shape[0], self.n_classes), dtype=np.float64)

            for i in range(self.n_classes):
                yp[:, i] = self.scds['%s' % i].predict(data, kind='vote', all=True).sum(axis=1)


            yp = yp.argmax(axis=1)
        return yp

    def bs(self, data, label):
        yp = np.zeros((data.shape[0], self.scds['0'].round, self.n_classes), dtype=np.float64)
        for i in range(self.n_classes):
            yp[:, :, i] = self.scds['%s' % i].predict(data, kind='vote', all=True)
        data_prob = yp.reshape((yp.shape[0], -1))
        self.mlp = MLPClassifier(hidden_layer_sizes=(30,), activation='relu', solver='sgd', alpha=0.0001,
                                    batch_size='auto',
                                    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000,
                                    shuffle=True,
                                    random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                                    nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-08, n_iter_no_change=10)
        self.mlp.fit(data_prob, label)


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer, load_wine
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    np.random.seed(2016826)
    # data = load_breast_cancer()
    data = load_wine()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.int8)
    train_data, test_data, train_label, test_label = train_test_split(x, y)

    scd = SCD(n_classes=3, nrows=0.2, nfeatures=1, w_inc1=0.17, w_inc2=0.2, tol=0.0000001,
              local_iter=100,
              num_iters=200, interval=10, round=32, updated_features=8,
              adaptive_inc=False, n_jobs=4, num_gpus=4, hidden_nodes=20,
              verbose=True, evaluation=True)

    a = time()
    scd.train(train_data, train_label, test_data, test_label)

    print('cost %.3f seconds' % (time() - a))
    yp = scd.predict(test_data)

    print('Accuracy: ',
          accuracy_score(y_true=train_label, y_pred=scd.predict(train_data)))
    print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
    yp = scd.predict(test_data, kind='vote')
    print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
    # output probability
    yp = scd.predict(test_data, kind='vote', all=True)
