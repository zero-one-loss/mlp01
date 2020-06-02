#!/bin/sh

#SBATCH -p datasci

#SBATCH --workdir= .
#SBATCH --job-name=imagenet_scd_attack

cd ..

gpu=0

for seed in 2019
do

#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001  --n_classes 10 \
#--train-size 200 --target imagenet_scd01_32_br05_nr075_ni500 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256

#
python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001  --n_classes 10 \
--train-size 200 --target imagenet_scd01mlp_32_br02_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256

#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001  --n_classes 10 \
#--train-size 200 --target imagenet_scd01mlp_32_br05_nr075_ni500_i1 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256
#
#
#python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001  --n_classes 10 \
#--train-size 200 --target imagenet_scd01mlp_32_br02_nr075_ni1000 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001  --n_classes 10 \
--train-size 200 --target imagenet_scd01mlp_32_br02_nr075_ni1000_i1 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256



python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target imagenet_svm_0001 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256 --n_classes 10

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target imagenet_mlp400_001 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256 --n_classes 10

python bb_attack.py --epsilon 0.0625 --Lambda 0.01 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
--train-size 200 --target imagenet_mlp1vall_20 --random-sign 1 --seed $seed --dataset imagenet --oracle-size 256 --n_classes 10

done
