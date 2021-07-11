import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool


data_path = '/home/soonshin/sss/dataset/ASC/2020task1b/TAU-urban-acoustic-scenes-2020-3class-development/'
csv_file = data_path + 'evaluation_setup/fold1_train.csv'
output_path = 'features/train_asc_3class_48k_logmel128'
feature_type = 'logmel'

sr = 48000
duration = 10
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft / 2)
num_time_bin = int(np.ceil(duration * sr / hop_length))

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()

for i in range(len(wavpath)):
    stereo, fs = sound.read(data_path + wavpath[i])
    logmel_data = np.zeros((num_freq_bin, num_time_bin, 1), 'float32')
    
    # mono-channel
    if stereo[0].shape == (2,):
        mono = []
        mono = (stereo[:,0] + stereo[:,1]) / 2
        
    # sampling rate
    if fs != sr:
        print ("sampling rate converted")
        mono = librosa.resample(mono, fs, sr, fix=True) 
        
    logmel_data[:,:,0] = librosa.feature.melspectrogram(mono[:], 
                                                        sr=sr, 
                                                        n_fft=num_fft, 
                                                        hop_length=hop_length,
                                                        n_mels=num_freq_bin, 
                                                        fmin=0.0, 
                                                        fmax=sr/2, 
                                                        htk=True, 
                                                        norm=None)
    logmel_data = np.log(logmel_data+1e-8)
    
    #for j in range(len(logmel_data[:,:,0][:,0])):
    #    mean = np.mean(logmel_data[:,:,0][j,:])
    #    std = np.std(logmel_data[:,:,0][j,:])
    #    logmel_data[:,:,0][j,:] = ((logmel_data[:,:,0][j,:]-mean)/std)
    #    logmel_data[:,:,0][np.isnan(logmel_data[:,:,0])]=0.

    feature_data = {'feat_data': logmel_data,}

    cur_file_name = output_path + wavpath[i][5:-3] + feature_type
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print (cur_file_name)
        
        

