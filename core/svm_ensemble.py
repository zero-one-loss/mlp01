from sklearn.svm import LinearSVC
import numpy as np
from torch.multiprocessing import Pool
from time import time

class SVM(object):
    def __init__(self, n_classes, round, **kwargs):
        self.clf = {}
        self.round = round
        self.n_classes = n_classes
        for i in range(round):
            self.clf[i] = LinearSVC(**kwargs)

    def parallel_fit(self, model, data, label):
        model.fit(data, label)
        return model

    def fit(self, train_data, train_label):

        a = time()
        pool = Pool(16)

        results = []
        for i in range(self.round):
            print(i)

            bag_index = np.random.choice(np.arange(train_label.shape[0]), train_label.shape[0])
            data = train_data[bag_index]
            label = train_label[bag_index]
            results.append(pool.apply_async(self.parallel_fit, args=(self.clf[i], data, label)))
        pool.close()
        pool.join()
        for i, result in enumerate(results):
            self.clf[i] = result.get()
        print('Class %d cost %.1f seconds' % (i, time() - a))

    def predict(self, test_data):
        yp = np.zeros((test_data.shape[0], self.round, self.n_classes))

        for j in range(self.round):
            yp[:, j] = self.clf[j].decision_function(test_data)
        yp = yp.sum(axis=1).argmax(axis=1)

        return yp