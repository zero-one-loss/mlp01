#!/bin/sh

#$-q datasci
#$-q datasci3
#$-q datasci4
#$-cwd
#$-N multi_node

cd ..




#python evaluate_corruption.py --dataset mnist --n_classes 10
#python evaluate_corruption.py --dataset cifar10 --n_classes 10
#python evaluate_corruption.py --dataset imagenet --n_classes 10
#
#
#python evaluate_fp.py --dataset cifar10 --n_classes 10
python evaluate_fp.py --dataset imagenet --n_classes 10