#!/bin/bash

hdf5_path='/home/soonshin/sss/workplace/etri-2021/exp/exp_combine17_0.4_logmel40_augs_mobilenet_ca_fusion_mobile/model-155-1.5356-0.6431.hdf5'
tflite_path='/home/soonshin/sss/workplace/etri-2021/exp/exp_combine17_0.4_logmel40_augs_mobilenet_ca_fusion_mobile/model-155-1.5356-0.6431.tflite'
save_path='/home/soonshin/sss/workplace/etri-2021/exp/exp_combine17_0.4_logmel40_augs_mobilenet_ca_fusion_mobile/model-155-1.5356-0.6431.csv'

python3 -W ignore model_quantize.py $hdf5_path $tflite_path
python3 -W ignore model_size.py $hdf5_path $tflite_path
python3 -W ignore eval_quantize.py $tflite_path $save_path