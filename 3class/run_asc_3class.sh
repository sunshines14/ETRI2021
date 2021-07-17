#!/bin/bash

hdf5_path='/home/soonshin/sss/workplace/etri-2021/3class/exp/exp_asc_3class_48k_logmel128_mobilenet_ca_fusion/model-183-0.3535-0.9388.hdf5'
tflite_path='/home/soonshin/sss/workplace/etri-2021/3class/exp/exp_asc_3class_48k_logmel128_mobilenet_ca_fusion/model-183-0.3535-0.9388.tflite'
save_path='/home/soonshin/sss/workplace/etri-2021/3class/exp/exp_asc_3class_48k_logmel128_mobilenet_ca_fusion/model-183-0.3535-0.9388.csv'

CUDA_VISIBLE_DEVICES=1 python3 -W ignore model_quantize.py $hdf5_path $tflite_path
CUDA_VISIBLE_DEVICES=1 python3 -W ignore model_size.py $hdf5_path $tflite_path
CUDA_VISIBLE_DEVICES=1 python3 -W ignore eval_quantize_asc_3class.py $tflite_path $save_path