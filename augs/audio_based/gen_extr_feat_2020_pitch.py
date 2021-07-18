import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool


overwrite = True

data_path = '/home/soonshin/sss/dataset/SED/'
csv_file = data_path + 'meta/train_combine17.csv'
output_path = '../../features/train_combine17_0.4_logmel40_norm_augs'
feature_type = 'logmel'
folder_name = data_path + 'train/'

sr = 16000
duration = 0.4
num_freq_bin = 40
num_fft = 321
hop_length = int(num_fft/2)
num_time_bin = int(np.ceil(duration*sr/hop_length))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep=',', encoding='ASCII')
wavpath = data_df['filenames'].tolist()

for i in range(len(wavpath)):
    stereo, fs = sound.read(folder_name + wavpath[i])
    n_step = np.random.uniform(-4, 4)
    y_pitched = librosa.effects.pitch_shift(stereo, sr, n_steps=n_step)
    length = len(stereo)
    stereo = y_pitched
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], 
                                                       sr=sr, 
                                                       n_fft=num_fft, 
                                                       hop_length=hop_length, 
                                                       n_mels=num_freq_bin, 
                                                       fmin=0.0, 
                                                       fmax=sr/2, 
                                                       htk=True, 
                                                       norm=None)

    logmel_data = np.log(logmel_data+1e-8)
    logmel_data = (logmel_data - np.min(logmel_data)) / (np.max(logmel_data) - np.min(logmel_data))

    feature_data = {'feat_data': logmel_data}

    cur_file_name = output_path + '/' + wavpath[i].replace('.wav','') + '_pitch.' + feature_type
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print(cur_file_name)