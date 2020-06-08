#!/bin/sh




cd download/cifar10-c

tar xvf CIFAR-10-C.tar

cd ../cifar10-p
tar xvf CIFAR-10-P.tar

cd download

cd imagenet-c

for folder in "n02037110" "n02128757" "n02808440" "n03459775" "n03956157" "n02051845" "n02128925" "n02814533" "n03461385" "n03958227"
do
  tar xvf blur.tar "*/*/$folder"
  tar xvf digital.tar "*/*/$folder"
  tar xvf extra.tar "*/*/$folder"
  tar xvf noise.tar "*/*/$folder"
  tar xvf weather.tar "*/*/$folder"
done

mv *.tar ../ic

cd ..

cd imagenet-p

for folder in "n02037110" "n02128757" "n02808440" "n03459775" "n03956157" "n02051845" "n02128925" "n02814533" "n03461385" "n03958227"
do
  tar xvf blur.tar "*/$folder"
  tar xvf digital.tar "*/$folder"
  tar xvf noise.tar "*/$folder"
  tar xvf weather.tar "*/$folder"
done

mv *.tar ../ip