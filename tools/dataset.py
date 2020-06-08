import os
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms

def get_data(data=None):
    if data == 'mnist':
        train_dir = '../data'
        test_dir = '../data'

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ])

        trainset = torchvision.datasets.MNIST(root=train_dir, train=True, download=True, transform=test_transform)
        testset = torchvision.datasets.MNIST(root=test_dir, train=False, download=True, transform=test_transform)

        save_path = '../data/mnist'

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        train_data = trainset.data.numpy()
        train_label = np.array(trainset.targets)
        test_data = testset.data.numpy()
        test_label = np.array(testset.targets)

        os.chdir(save_path)
        np.save('train_image.npy', train)
        np.save('train_label.npy', train_label)
        np.save('test_image.npy', test)
        np.save('test_label.npy', test_label)
        os.chdir(os.pardir)


    elif data == 'cifar10':

        train_dir = '../data'
        test_dir = '../data'

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ])

        trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=test_transform)

        testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True, transform=test_transform)

        save_path = '../data/cifar10'

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        os.chdir(save_path)
        np.save('train_image.npy', trainset.data)
        np.save('train_label.npy', np.array(trainset.targets))
        np.save('test_image.npy', testset.data)
        np.save('test_label.npy', np.array(testset.targets))
        os.chdir(os.pardir)

    elif data == 'stl10':
        pass

    elif data == 'imagenet':
        train_dir = '../data/sub_imagenet/train_mc'
        test_dir = '../data/sub_imagenet/val_mc'

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ])

        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False,
                                                   num_workers=16, pin_memory=True, drop_last=True)

        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False,
                                                  num_workers=10, pin_memory=True)

        train_data = []
        test_data = []
        train_labels = []
        test_labels = []

        for idx, (data, targets) in enumerate(train_loader):
            print(idx)
            train_data.append(data.numpy())
            train_labels.append(targets.numpy())

        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        for idx, (data, targets) in enumerate(test_loader):
            print(idx)
            test_data.append(data.numpy())
            test_labels.append(targets.numpy())

        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        save_path = '../data/imagenet'

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        os.chdir(save_path)
        np.save('train_image.npy', train_data)
        np.save('train_label.npy', train_labels)
        np.save('test_image.npy', test_data)
        np.save('test_label.npy', test_labels)
        os.chdir(os.pardir)


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