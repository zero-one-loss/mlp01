#!/bin/sh



cd download/cifar10-p

wget -c https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
wget -c https://zenodo.org/record/2535967/files/CIFAR-10-P.tar

cd ../imagenet-c

wget -c https://zenodo.org/record/2235448/files/blur.tar
wget -c https://zenodo.org/record/2235448/files/digital.tar
wget -c https://zenodo.org/record/2235448/files/extra.tar
wget -c https://zenodo.org/record/2235448/files/noise.tar
wget -c https://zenodo.org/record/2235448/files/weather.tar

cd ../imagenet-p

wget -c https://zenodo.org/record/3565846/files/blur.tar
wget -c https://zenodo.org/record/3565846/files/digital.tar
wget -c https://zenodo.org/record/3565846/files/noise.tar
wget -c https://zenodo.org/record/3565846/files/weather.tar



