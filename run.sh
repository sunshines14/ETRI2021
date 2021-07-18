#!/bin/bash

hdf5_path='/home/soonshin/sss/workplace/etri-2021/exp/exp_combine11_0.4_logmel40_norm_augs_mobilenet_ca_fusion/model-047-0.7254-0.8455.hdf5'
tflite_path='/home/soonshin/sss/workplace/etri-2021/exp/exp_combine11_0.4_logmel40_norm_augs_mobilenet_ca_fusion/model-047-0.7254-0.8455.tflite'
save_path='/home/soonshin/sss/workplace/etri-2021/exp/exp_combine11_0.4_logmel40_norm_augs_mobilenet_ca_fusion/model-047-0.7254-0.8455.csv'

python3 -W ignore model_quantize.py $hdf5_path $tflite_path
python3 -W ignore model_size.py $hdf5_path $tflite_path
python3 -W ignore eval_quantize.py $tflite_path $save_path