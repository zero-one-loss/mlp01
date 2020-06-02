import os
import numpy as np



def load_data(data=None, n_classes=2):
    
    if data == 'mnist':
        curdir = os.getcwd()
        os.chdir('../data/mnist')
        train_data = np.load('train_image.npy').reshape((-1, 28 * 28)) / 255
        test_data = np.load('test_image.npy').reshape((-1, 28 * 28)) / 255
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)
        

        
    elif data == 'cifar10':
        curdir = os.getcwd()
        os.chdir('../data/cifar10')
        train_data = np.load('train_image.npy').reshape((-1, 32 * 32 * 3)) / 255
        test_data = np.load('test_image.npy').reshape((-1, 32 * 32 * 3)) / 255
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'stl10':
        curdir = os.getcwd()
        os.chdir('../data/stl10')
        train_data = np.load('train_image.npy').reshape((-1, 96 * 96 * 3)) / 255
        test_data = np.load('test_image.npy').reshape((-1, 96 * 96 * 3)) / 255
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'imagenet':
        curdir = os.getcwd()
        os.chdir('../data/imagenet')
        train_data = np.load('train_image.npy').reshape((-1, 224 * 224 * 3))
        test_data = np.load('test_image.npy').reshape((-1, 224 * 224 * 3)) 
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)


    else:
        raise AssertionError("%s is not in the list" % data)

    train = train_data[train_label < n_classes]
    test = test_data[test_label < n_classes]
    train_label = train_label[train_label < n_classes]
    test_label = test_label[test_label < n_classes]

    return train, test, train_label, test_label


if __name__ == '__main__':

    train, test, train_label, test_label = load_data('mnist')