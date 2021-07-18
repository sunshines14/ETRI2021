#!/bin/bash

hdf5_path='/home/soonshin/sss/workplace/etri-2021/3class/exp/exp_asc_3class_48k_logmel128_norm_mobilenet_ca_fusion/model-215-0.3754-0.9364.hdf5'
tflite_path='/home/soonshin/sss/workplace/etri-2021/3class/exp/exp_asc_3class_48k_logmel128_norm_mobilenet_ca_fusion/model-215-0.3754-0.9364.tflite'
save_path='/home/soonshin/sss/workplace/etri-2021/3class/exp/exp_asc_3class_48k_logmel128_norm_mobilenet_ca_fusion/model-215-0.3754-0.9364.csv'

python3 -W ignore model_quantize.py $hdf5_path $tflite_path
python3 -W ignore model_size.py $hdf5_path $tflite_path
python3 -W ignore eval_quantize_asc_3class.py $tflite_path $save_path