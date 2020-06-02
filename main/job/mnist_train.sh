#!/bin/sh

#$-q datasci
#$-q datasci3
#$-q datasci4
#$-cwd
#$-N scd_mnist

cd ..

# scd01

# 10 015 0.5
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br05_nr015_ni10.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br05_nr015_ni10.txt

# 10 075 0.5
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br05_nr075_ni10.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br05_nr075_ni10.txt

# 10 015 0.2
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br02_nr015_ni10.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br02_nr015_ni10.txt

# 10 075 0.2
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 10 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br02_nr075_ni10.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br02_nr075_ni10.txt



# 100 015 0.5
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br05_nr015_ni100.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br05_nr015_ni100.txt

# 100 075 0.5
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br05_nr075_ni100.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br05_nr075_ni100.txt

# 100 015 0.2
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br02_nr015_ni100.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br02_nr015_ni100.txt

# 100 075 0.2
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 100 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br02_nr075_ni100.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br02_nr075_ni100.txt


# 500 015 0.5
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc 0.17 --num-iters 500 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br05_nr015_ni500.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br05_nr015_ni500.txt

# 500 075 0.5
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 500 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br05_nr075_ni500.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br05_nr075_ni500.txt

# 500 015 0.2
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc 0.17 --num-iters 500 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br02_nr015_ni500.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br02_nr015_ni500.txt

# 500 075 0.2
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 500 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br02_nr075_ni500.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br02_nr075_ni500.txt


# 1000 015 0.5
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br05_nr015_ni1000.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br05_nr015_ni1000.txt

# 1000 075 0.5
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br05_nr075_ni1000.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br05_nr075_ni1000.txt

# 1000 015 0.2
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br02_nr015_ni1000.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br02_nr015_ni1000.txt

# 1000 075 0.2
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01_32_br02_nr075_ni1000.pkl --dataset mnist --version linear --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01_32_br02_nr075_ni1000.txt


# scd01mlp

# 100 015 0.5
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 100 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br05_nr015_ni100.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br05_nr015_ni100.txt

# 100 075 0.5
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 100 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br05_nr075_ni100.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br05_nr075_ni100.txt

# 100 015 0.2
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 100 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br02_nr015_ni100.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br02_nr015_ni100.txt

# 100 075 0.2
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 100 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br02_nr075_ni100.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br02_nr075_ni100.txt


# 500 015 0.5
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 500 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br05_nr015_ni500.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br05_nr015_ni500.txt

# 500 075 0.5
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 500 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br05_nr075_ni500.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br05_nr075_ni500.txt

# 500 015 0.2
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 500 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br02_nr015_ni500.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br02_nr015_ni500.txt

# 500 075 0.2
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 500 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br02_nr075_ni500.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br02_nr075_ni500.txt


# 1000 015 0.5
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br05_nr015_ni1000.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br05_nr015_ni1000.txt

# 1000 075 0.5
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 --b-ratio 0.5 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br05_nr075_ni1000.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br05_nr075_ni1000.txt

# 1000 015 0.2
python train_scd.py --nrows 0.15 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br02_nr015_ni1000.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br02_nr015_ni1000.txt

# 1000 075 0.2
python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 --b-ratio 0.2 \
--updated-features 64 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 10 \
--target mnist_scd01mlp_32_br02_nr075_ni1000.pkl --dataset mnist --version mlp --seed 2018 --width 10000 \
--metrics balanced --init normal --verbose > logs/mnist/mnist_scd01mlp_32_br02_nr075_ni1000.txt
