from sklearn.neural_network import MLPClassifier
import numpy as np
from torch.multiprocessing import Pool
from time import time

class MLP(object):
    def __init__(self, n_classes, round, **kwargs):
        self.clf = {}
        self.round = round
        self.n_classes = n_classes
        for i in range(n_classes):
            self.clf[i] = [MLPClassifier(**kwargs)] * round

    def parallel_fit(self, model, data, label):
        model.fit(data, label)
        return model

    def fit(self, train_data, train_label):

        for i in range(self.n_classes):
            a = time()

            print(i)

            temp_label = (train_label == i).astype(np.int32)
            for j in range(self.round):
                print(j)
                bag_index = np.random.choice(np.arange(temp_label.shape[0]), temp_label.shape[0])
                data = train_data[bag_index]
                label = temp_label[bag_index]

                self.clf[i][j].fit(data, label)

            print('Class %d cost %.1f seconds' % (i, time() - a))
    def predict(self, test_data):
        yp = np.zeros((test_data.shape[0], self.round, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.round):
                yp[:, j, i] = self.clf[i][j].predict_proba(test_data)[:, 1]
                # yp[:, j, i] = self.clf[i][j].predict(test_data)
        yp = yp.sum(axis=1).argmax(axis=1)

        return yp