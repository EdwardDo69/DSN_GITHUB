#!/bin/bash
save_dir="./save_model/train_adap_car_30epoch"
dataset="mskda_car"
net="vgg16"
pretrained_path="./pre_trained_model/vgg16_caffe.pth"
max_epoch=30
burn_in=10

CUDA_VISIBLE_DEVICES=3 python train_msda.py --cuda --dataset ${dataset} \
--net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --max_epoch ${max_epoch} --burn_in ${burn_in} \
#>train_msda_car_30epoch.txt >&1
