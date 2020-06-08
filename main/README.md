main
=

Introduction
-
This directory contains experiment scripts.
If you want to reproduce the experiment results, 
please follow steps below.

1. Preparing data

    Experiments include three benchmarks data set. MNIST, CIFAR10, IMAGENET.
    
    MNIST data set is downloaded through `torchvision`. You can go to `tools`, 
    run `get_data('mnist')` in `dataset.py`. It will download data into `data`
    directory.
    
    CIFAR10 data set is downloaded through `torchvision`. You can go to `tools`, 
    run `get_data('cifar10')` in `dataset.py`. It will download data into `data`
    directory.
    
    IMAGENET data set should be prepared by yourself. Because we only use 10 classes
    from the full dataset and split them into training and testing manually, we 
    provide file name list `train_mc` and `val_mc` in `data` to help you splitting them. Please modify the code `get_data` in `tools`
    to access your imagenet data path.
    
    Corruption and purturbation data set you can collect them through `download_corruption.sh`
    and `up.sh` in `data` directory.
    
2. Training SCD01 and MLP01.
     
   Training scripts are in `job`, their name are `mnist_train.sh`, `cifar10_train.sh`, and
   `imagenet_train.sh`.
   
   Blackbox attack scripts are in `job`, their name are 
   Training scripts are in `job`, their name are `mnist_blackbox_attack.sh`, `cifar10_blackbox_attack.sh`, and
   `imagenet_blackbox_attack.sh`.
   
3. Evaluating corruption and purturbation results.

    You need to modify the model name which your just saved after training for 
    SCD01 or MLP01 in `evaulate_corruption.py`. Then you can run `corru.sh` in `job`.
    