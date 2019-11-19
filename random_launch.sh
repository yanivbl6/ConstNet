#!/bin/bash

dataset=$1

arch="network_4x5"
arch="16_lr_30_dropout_15"

while [[ $2 ]]; do
    rate=$2

    dev1=`nvidia-smi | grep python | awk '{print $2}' | grep 1 | wc -l`
    dev3=`nvidia-smi | grep python | awk '{print $2}' | grep 3`


    if [[ $dev1 -eq 0 ]]; then
        dev="0:1"
    elif [[ $dev3 -eq 0 ]]; then
        dev="2:3"
    else
        echo "No available devices"
        dev="0:1"
    fi

    name=${dataset}_random_${rate}

    if [[ -d "runs/$name" ]]; then
        echo "Directory ${name} already exists"
        exit
    fi

    cmd="python train.py --layers 16 --widen-factor 10 --fixup --batchnorm --lr 0.03 --name $name -d ${dev} --droprate 0.15 --prune ${dataset}_constnet_${arch} --cutoff 0.${rate} --prune_epoch 0 --dataset ${dataset} --no-saves --randomize_mask "

    echo "Running command:"
    echo $cmd
    eval $cmd


    shift 1
done
