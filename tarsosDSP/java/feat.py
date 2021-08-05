import os
import struct
import numpy as np
from scipy import signal
from multiprocessing import Process

import utils
import table
import config


def gen_sox_script(in_dir, out_dir, in_opt, out_opt, in_ext, out_ext, cmdfile):
    print(in_dir)
    print(out_dir)

    with open(cmdfile, 'w') as f:
        for (path, dir, files) in os.walk(in_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == in_ext:
                    key = utils.gen_key_from_file(filename)
                    infile = in_dir.rstrip('/')+'/'+filename
                    outfile = out_dir.rstrip('/')+'/'+key+out_ext
                    cmd = 'sox' + ' ' + in_opt + ' ' + infile + ' ' + out_opt + ' ' + outfile + '\n'
                    f.write(cmd)
                    cmd = 'echo ' + outfile + '\n'
                    f.write(cmd)


def gen_hcopy_script(in_dir, out_dir, in_ext, cmdfile, hCopy_feat, hCopy_feat_txt):
    print(in_dir)
    print(out_dir)

    with open(cmdfile, 'w') as f:
        for (path, dir, files) in os.walk(in_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == in_ext:
                    key = utils.gen_key_from_file(filename)
                    infile = in_dir.rstrip('/') + '/' + filename
                    fbank_file = out_dir.rstrip('/') + '/' + key + hCopy_feat
                    fbtxt_file = out_dir.rstrip('/') + '/' + key + hCopy_feat_txt
                    cmd = 'HCopy -C HCopy.config ' + infile + ' ' + fbank_file + '\n'
                    f.write(cmd)
                    cmd = 'HList -r ' + fbank_file + ' > ' + fbtxt_file + '\n'
                    f.write(cmd)
                    cmd = 'echo ' + fbtxt_file + '\n'
                    f.write(cmd)


def spectrogram(filename, samplerate, nperseg, noverlap, nfft, return_onesided=True):
    file_size = os.path.getsize(filename)
    n_sample = file_size / 2

    f = open(filename, 'rb')
    binary_data = f.read(file_size)
    h = ''
    for i in range(0, n_sample):
        h += 'h'
    raw_pcm = struct.unpack(h, binary_data)
    raw_pcm_np = np.asarray(raw_pcm, dtype=np.float32)

    f, t, spec = signal.spectrogram(raw_pcm_np, fs=samplerate, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                    return_onesided=return_onesided)

    return f, t, spec


def consecutive_windows_data(data, n_input_frame, nshift=1):
    (data_length, data_dimension) = data.shape
    if data_length < config.N_INPUT_FRAME:
        return None
    else:
        #cw_data = np.zeros(((data_length - n_input_frame)/nshift, n_input_frame * data_dimension))
        nImage = (data_length - n_input_frame) / nshift + 1
        nPixel = n_input_frame * data_dimension
        cw_data = np.zeros((nImage, nPixel))
        #for i in range(0, (data_length-n_input_frame)/nshift):
        for i in range(0, nImage):
            cw_data[i] = data[i*nshift:i*nshift+n_input_frame].reshape(-1)
        return cw_data


def consecutive_windows_data_sum(data, n_input_frame):
    (data_length, data_dimension) = data.shape
    cw_data = np.zeros((data_length-n_input_frame, data_dimension))

    for i in range(0, data_length-n_input_frame):
        cw_data[i] = data[i:i+n_input_frame].sum(axis=0)
    return cw_data


def consecutive_windows_data_feat_delta(data):
    (data_length, data_dimension) = data.shape
    delta_data = np.zeros((data_length, data_dimension+data_dimension-1))

    for i in range(0, data_length):
        delta_data[i][0:data_dimension] = data[i]
        for j in range(0, data_dimension-1):
            delta_data[i][data_dimension+j] = data[i][j+1] - data[i][j]
    return delta_data


def extract_feature_func(file_list, feature_type='spectrogram', gtruthtable=None):

    for (infile, outfile) in file_list:

        """ To extract difference features, modify code below  """

        if feature_type is 'spectrogram':
            print(infile)
            _, _, spec = spectrogram(filename=infile,
                                     samplerate=44100, nperseg=512, noverlap=71, nfft=512, return_onesided=True)
            np.save(outfile, spec)
        elif feature_type is 'txt2np':
            data = np.loadtxt(infile)
            np.save(outfile, data)
        elif feature_type is 'txt2np_consecutive_windows':
            data = np.loadtxt(infile, delimiter=', ')
            key = utils.gen_key_from_file(infile)
            cur_id = int(gtruthtable.key2id(key))
            if os.path.isfile(outfile) or cur_id < 0:
                continue

            n_shift = config.n_frame_shift_PersonalMedia_30class[cur_id]
            # print n_shift
            cw_data = consecutive_windows_data(data=data, n_input_frame=config.N_INPUT_FRAME, nshift=n_shift)
            if cw_data is not None:
                np.save(outfile, cw_data)
        elif feature_type is 'txt2np_consecutive_windows_sum':
            data = np.loadtxt(infile)
            cw_data = consecutive_windows_data_sum(data=data, n_input_frame=config.N_INPUT_FRAME)
            if cw_data is not None:
                np.save(outfile, cw_data)
        elif feature_type is 'txt2np_consecutive_windows_feat_delta':
            data = np.loadtxt(infile)
            cw_data = consecutive_windows_data_feat_delta(data=data)
            if cw_data is not None:
                np.save(outfile, cw_data)

        """ To extract difference features, modify code above  """

def extract_feature(in_dir, out_dir, in_ext, out_ext, feature_type='spectrogram', gtruthtable=None, nj=1):
    print(in_dir)
    print(out_dir)

    filelist = []

    for (path, dir, files) in os.walk(in_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == in_ext:
                key = utils.gen_key_from_file(filename)
                infile = in_dir.rstrip('/') + '/' + filename
                outfile = out_dir.rstrip('/') + '/' + key + out_ext
                cur_id = int(ground_truth_table.key2id(key))
                if os.path.isfile(outfile+'.npy') or cur_id < 0:
                    print('[EXIST]'+'['+str(cur_id)+']'+outfile)
                    continue
                filelist.append([infile, outfile])

    filelistlen = len(filelist)
    #print(filelistlen)
    filelistsplit = [filelist[i*filelistlen/nj:min((i+1)*filelistlen/nj, filelistlen)]
                     for i in xrange(0, nj)]
    #print(filelistsplit)
    pList = []
    for i in range(0, nj):
        pList.append(Process(target=extract_feature_func, args=(filelistsplit[i], feature_type, gtruthtable,)))
        pList[i].start()
    for i in range(0, nj):
        pList[i].join()
    print(pList)

    return


def gen_numpy_data_from_list(gtruthfile, listfile, out_image_file, out_label_file, label=True, shuffle=False):
    if label is True:
        ground_truth_table = table.GroundTruthTable(config.class_list_PersonalMedia_30class, gtruthfile)
    
    len_total_data = 0
    size_data = 0

    """ First pass for calculating total size of numpy data """
    with open(listfile) as f:
        for line in f.readlines():
            print('1: ' + line)
            key = utils.gen_key_from_file(line)
            if label is True:
                try:
                    class_id = ground_truth_table.key2id(key)
                except:
                    continue
            data_file_path = line.strip()
            cur_data = np.load(data_file_path)
            if len_total_data == 0:
                size_data = cur_data.shape[1]
            len_total_data += cur_data.shape[0]
            # print data_file_path + ' ' + str(cur_data.shape[0])

    print('Total length of data: ' + str(len_total_data))
    total_image_data = np.zeros((len_total_data, size_data))
    if label is True:
        total_label_data = np.zeros((len_total_data, 1))

    """ Second pass for loading data """
    with open(listfile) as f:
        cur_idx = 0
        for line in f.readlines():
            print('2: ' + line)
            data_file_path = line.strip()
            cur_data = np.load(data_file_path)
            cur_len = cur_data.shape[0]
            if cur_len == 0:
                continue
            total_image_data[cur_idx:cur_idx+cur_len] = cur_data

            key = utils.gen_key_from_file(line)

            if label is True:
                try:
                    class_id = ground_truth_table.key2id(key)
                except:
                    continue
                total_label_data[cur_idx:cur_idx+cur_len] = class_id
            cur_idx += cur_len
            # print data_file_path + ' ' + str(cur_data.shape[0])

    if shuffle is True:
        assert(label is True)
        tmp_shuffle = np.hstack([total_image_data, total_label_data])
        np.random.shuffle(tmp_shuffle)
        tmp_shuffle_tr = tmp_shuffle.transpose()
        total_image_data = tmp_shuffle_tr[:-1].transpose()
        total_label_data = tmp_shuffle_tr[-1:].transpose()

    np.save(out_image_file, total_image_data)
    if label is True:
        np.save(out_label_file, total_label_data.reshape(len_total_data))

    return


if __name__ == "__main__":

    #from wav (or TarsosDSP)
    audio_format_transition = False
    hCopy_command_execution = False
    hCopy_feat = '.fb40'
    hCopy_feat_txt = '.fb40txt'

    #make npy
    feature_extraction = True 
    additional_ext = '.cw40.vshift'

    #make one npy
    feature_concatenation = False 
    njob = 1

    if audio_format_transition is True:
        #FREESOUND
        # gen_sox_script(
        #     in_dir='/home/ubuntu/data/data/FREESOUND/segmented',
        #     out_dir='/home/ubuntu/data/work/personal_30class/data/wav/freesound',
        #     in_opt='-r 16000 -e signed -b 16 -c 1 -t raw',
        #     out_opt='-r 16000 -e signed -b 16 -c 1 -t nist',
        #     in_ext='.pcm', out_ext='.wav', cmdfile='./cmd')
        # os.system('chmod +x ./cmd')
        # os.system('./cmd')

        UrbanSound8K
        gen_sox_script(
            in_dir='/home/splab/DATA/DCASE/DCASE2017/task1/TUT-acoustic-scenes-2017-development/audio',
            out_dir='/home/splab/PersonalMedia/tf/DCASE2017/data/fbank/dev',
            in_opt='-t wav',
            out_opt='-r 16000 -e signed -b 16 -c 1 -t nist',
            in_ext='.wav', out_ext='.wav', cmdfile='./cmd')
        os.system('chmod +x ./cmd')
        os.system('./cmd')

        #BBCSoundFX
        # gen_sox_script(
        #     in_dir='/home/ubuntu/data/data/BBCSOUNDFX/BBC_cut',
        #     out_dir='/home/ubuntu/data/work/personal_30class/data/wav/bbcsoundfx',
        #     in_opt='-r 44100 -e signed -b 16 -c 1 -t raw',
        #     out_opt='-r 16000 -e signed -b 16 -c 1 -t nist',
        #     in_ext='.pcm', out_ext='.wav', cmdfile='./cmd')
        # os.system('chmod +x ./cmd')
        # os.system('./cmd')

        #DCASE2016
        # gen_sox_script(
        #     in_dir='/home/ubuntu/data/data/DCASE2016/task1/TUT-acoustic-scenes-2016-development/audio',
        #     out_dir='/home/ubuntu/data/work/personal_30class/data/wav/dcase2016',
        #     in_opt='-t wav',
        #     out_opt='-r 16000 -e signed -b 16 -c 1 -t nist',
        #     in_ext='.wav', out_ext='.wav', cmdfile='./cmd')
        # os.system('chmod +x ./cmd')
        # os.system('./cmd')

    if hCopy_command_execution is True:
        # gen_hcopy_script(
        #     in_dir='/home/ubuntu/data/work/personal_30class/data/wav/freesound',
        #     out_dir='/home/ubuntu/data/work/personal_30class/data/hcopy/freesound',
        #     in_ext='.wav',
        #     cmdfile='./hcopy', hCopy_feat=hCopy_feat, hCopy_feat_txt=hCopy_feat_txt)
        # os.system('chmod +x ./hcopy')
        # os.system('./hcopy')

        gen_hcopy_script(
            in_dir='/home/splab/PersonalMedia/tf/DCASE2017/data/fbank/dev',
            out_dir='/home/splab/PersonalMedia/tf/DCASE2017/data/fbank/dev',
            in_ext='.wav',
            cmdfile='./hcopy', hCopy_feat=hCopy_feat, hCopy_feat_txt=hCopy_feat_txt)
        os.system('chmod +x ./hcopy')
        os.system('./hcopy')

        # gen_hcopy_script(
        #     in_dir='/home/ubuntu/data/work/personal_30class/data/wav/bbcsoundfx',
        #     out_dir='/home/ubuntu/data/work/personal_30class/data/hcopy/bbcsoundfx',
        #     in_ext='.wav',
        #     cmdfile='./hcopy', hCopy_feat=hCopy_feat, hCopy_feat_txt=hCopy_feat_txt)
        # os.system('chmod +x ./hcopy')
        # os.system('./hcopy')
        #
        # gen_hcopy_script(
        #     in_dir='/home/ubuntu/data/work/personal_30class/data/wav/dcase2016',
        #     out_dir='/home/ubuntu/data/work/personal_30class/data/hcopy/dcase2016',
        #     in_ext='.wav',
        #     cmdfile='./hcopy', hCopy_feat=hCopy_feat, hCopy_feat_txt=hCopy_feat_txt)
        # os.system('chmod +x ./hcopy')
        # os.system('./hcopy')

    if feature_extraction is True:
        gtruthfile = 'ground_truth/example'
        ground_truth_table = table.GroundTruthTable(config.class_list_PersonalMedia_30class, gtruthfile)
        
        PATH = 'feat'
        
        #class_dir_list = os.listdir(PATH)
        
        #for class_dir in class_dir_list:
        #    feature_path = PATH + '/' + class_dir

        #    extract_feature(
        #        in_dir=feature_path,
        #        out_dir=feature_path,
        #        in_ext=hCopy_feat_txt, out_ext=hCopy_feat_txt+additional_ext, feature_type='txt2np_consecutive_windows',
        #        gtruthtable=ground_truth_table,
        #        nj=1)
        
        extract_feature(
            in_dir=PATH,
            out_dir=PATH,
            in_ext=hCopy_feat_txt, out_ext=hCopy_feat_txt+additional_ext, feature_type='txt2np_consecutive_windows',
            gtruthtable=ground_truth_table,
            nj=1)

    if feature_concatenation is True:
        gtruthfile = 'ground_truth/etri2020'
        train_path = 'train.txt'
        val_path = 'validation.txt'
        test_mobile_path = 'test_mobile.txt'
        test_nomobile_path = 'test_nomobile.txt'

        gen_numpy_data_from_list(
            gtruthfile, train_path, 
            'npy/image_tr'+hCopy_feat+additional_ext,
            'npy/label_tr'+hCopy_feat+additional_ext,
            label=True, shuffle=True)

        gen_numpy_data_from_list(
            gtruthfile, val_path,
            'npy/image_va'+ hCopy_feat + additional_ext,
            'npy/label_va' + hCopy_feat + additional_ext,
            label=True, shuffle=True)

        gen_numpy_data_from_list(
            gtruthfile, test_mobile_path,
            'npy/image_te_mo' + hCopy_feat + additional_ext,
            'npy/label_te_mo' + hCopy_feat + additional_ext,
            label=True, shuffle=True)

        gen_numpy_data_from_list(
            gtruthfile, test_nomobile_path,
            'npy/image_te_nomo' + hCopy_feat + additional_ext,
            'npy/label_te_nomo' + hCopy_feat + additional_ext,
            label=True, shuffle=True)

        # gen_numpy_data_from_list(
        #     '/home/splab/DATA/DCASE/DCASE2016/task1/TUT-acoustic-scenes-2016-development/evaluation_setup/fold1_evaluate.txt',
        #     '/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/dev',
        #     hCopy_feat_txt+additional_ext+'.npy',
        #     '/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/image_dev_fold1_eval'+hCopy_feat+additional_ext,
        #     '/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/label_dev_fold1_eval'+hCopy_feat+additional_ext,
        #     label=True, shuffle=True)
        #
        # gen_numpy_data_from_list(
        #     '/home/splab/DATA/DCASE/DCASE2016/task1/TUT-acoustic-scenes-2016-development/evaluation_setup/fold1_test.txt',
        #     '/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/dev',
        #     hCopy_feat_txt+additional_ext+'.npy',
        #     '/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/image_dev_fold1_test'+hCopy_feat+additional_ext,
        #     '/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/label_dev_fold1_test'+hCopy_feat+additional_ext,
        #     label=False, shuffle=False)

        # gen_numpy_data_from_list(
        #     list_file=
        #     '/home/splab/DATA/DCASE/DCASE2016/task1/TUT-acoustic-scenes-2016-evaluation/evaluation_setup/test.txt',
        #     in_dir='/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/eval',
        #     in_ext=hCopy_feat_txt+additional_ext+'.npy',
        #     out_image_file='/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/image_eval' +hCopy_feat+additional_ext,
        #     out_label_file='/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/label_eval' +hCopy_feat+additional_ext,
        #     label=False, shuffle=False)

        # gen_numpy_data_from_list(
        #     list_file=
        #     '/home/splab/DATA/DCASE/DCASE2016/task1/TUT-acoustic-scenes-2016-development/meta.txt',
        #     in_dir='/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/dev',
        #     in_ext=hCopy_feat_txt+additional_ext+'.npy',
        #     out_image_file='/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/image_train'+hCopy_feat+additional_ext,
        #     out_label_file='/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r16000_c1_b16/label_train'+hCopy_feat+additional_ext,
        #     label=True, shuffle=True)
