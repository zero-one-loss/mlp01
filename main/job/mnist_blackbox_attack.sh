#!/bin/sh


cd ..

gpu=0

for seed in 2019
do

python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
--train-size 200 --target mnist_scd01_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10


python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
--train-size 200 --target mnist_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10


python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
--train-size 200 --target mnist_svm --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10

python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
--train-size 200 --target mnist_mlp --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10


python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
--train-size 200 --target mnist_mlp1vall --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10

python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
--train-size 200 --target mnist_svm1vall_32vote --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10

python bb_attack.py --epsilon 0.2 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.001 --n_classes 10 \
--train-size 200 --target mnist_mlp1vall_32vote --random-sign 1 --seed $seed --dataset mnist --oracle-size 1024 --n_classes 10

done
