#!/bin/sh


cd ..

# scd01


# 1000 075 0.2
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 500 --b-ratio 0.2 --iters 1 \
--updated-features 256 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target imagenet_scd01_32_br02_nr075_ni500_i1.pkl --dataset imagenet --version linear --seed 2018 --width 1000 \
--metrics balanced --init normal --verbose > logs/imagenet/imagenet_scd01_32_br02_nr075_ni500_i1.txt


# scd01mlp


# 1000 075 0.2
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 --b-ratio 0.2 \
--updated-features 256 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 --iters 1 \
--target imagenet_scd01mlp_32_br02_nr075_ni1000_i1.pkl --dataset imagenet --version mlp --seed 2018 --width 1000 \
--metrics balanced --init normal --verbose > logs/imagenet/imagenet_scd01mlp_32_br02_nr075_ni1000_i1.txt
