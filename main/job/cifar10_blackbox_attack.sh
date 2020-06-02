#!/bin/sh

#SBATCH -p datasci3

#SBATCH --workdir= .
#SBATCH --job-name=cifar10_scd_attack

cd ..

gpu=1

for seed in 2019
do


#
python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target cifar10_scd01_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset cifar10 \
--oracle-size 1024 --n_classes 10

#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br05_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target cifar10_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset cifar10 \
--oracle-size 1024 --n_classes 10
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_scd01mlp_32_br05_nr075_ni1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 10


python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target cifar10_svm --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target cifar10_mlp --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10



python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target cifar10_mlp1vall --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target cifar10_svm1vall_32vote --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10

python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target cifar10_mlp1vall_32vote --random-sign 1 --seed $seed --dataset cifar10 --oracle-size 1024 --n_classes 10

done
