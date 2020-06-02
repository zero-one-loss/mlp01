import numpy as np
import pickle
import sys
sys.path.append('..')
from core.mlp_ensemble import MLP
import time
import os
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from tools import args, save_checkpoint, print_title, load_data

et, vc = print_title()

train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

np.random.seed(2018)



print('training data size: ')
print(train.shape)
print('testing data size: ')
print(test.shape)







np.random.seed(2018)

scd = MLP(n_classes=10, round=args.round, hidden_layer_sizes=(args.hidden_nodes, ), activation='logistic', solver='sgd',
          alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=args.lr, power_t=0.5, max_iter=args.iters,
          shuffle=True,
                    random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                    nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, n_iter_no_change=10)
a = time.time()
scd.fit(train, train_label)
print('Cost: %.3f seconds'%(time.time() - a))
yp = scd.predict(test)

print('Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
print('Test Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))

save_path = 'checkpoints'
save_checkpoint(scd, save_path, args.target, et, vc)



