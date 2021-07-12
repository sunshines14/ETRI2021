import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from utils_asc_3class import *
import sklearn.metrics as metrics
from plots import plot_confusion_matrix

#=========================================================================================================#
def evaluate_model(interpreter, test_images, test_labels, num_class, is_eval=False):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    prediction_digits = []
    pred_output_all = np.empty([1, num_class])
    for test_image in test_images:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)
        
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_index)
        pred_output = output[0]
        pred_output.reshape([1, num_class])
        pred_output_all = np.vstack((pred_output_all, pred_output))
        digit = np.argmax(output[0])
        prediction_digits.append(digit)
             
    pred_output_all = pred_output_all[1:,:]
    
    if is_eval:
        return pred_output_all, prediction_digits
    else:
        accurate_count = 0
        for index in range(len(prediction_digits)):
            if prediction_digits[index] == test_labels[index]:
                accurate_count += 1
        accuracy = accurate_count * 1.0 / len(prediction_digits)
        return accuracy, pred_output_all, prediction_digits

#=========================================================================================================#
is_eval = False

if not is_eval:
    data_path = '/home/soonshin/sss/dataset/ASC/2020task1b/TAU-urban-acoustic-scenes-2020-3class-development/'
    val_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
    feat_path = 'features/valid_asc_3class_48k_logmel128_norm'
    model_path = sys.argv[1]
    csv_path = sys.argv[2].replace('.csv','-asc-3class.csv')
    
else:
    data_path = '/home/soonshin/sss/dataset/ASC/2020task1b/TAU-urban-acoustic-scenes-2020-3class-development/'
    val_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
    feat_path = 'features/valid_asc_3class_48k_logmel128_norm'
    model_path = sys.argv[1]
    csv_path = sys.argv[2].replace('.csv','-eval.csv')

num_freq_bin = 128
num_classes = 3

print (model_path)
print (csv_path)

#=========================================================================================================#
if not is_eval:
    data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
    data_deltas_val = deltas(data_val)
    data_deltas_deltas_val = deltas(data_deltas_val)
    data_val = np.concatenate((data_val[:,:,4:-4,:], data_deltas_val[:,:,2:-2,:], data_deltas_deltas_val), axis=-1)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)
    print(data_val.shape)
    print(y_val.shape)
    
    dev_test_df = pd.read_csv(val_csv, sep='\t', encoding='ASCII')
    wav_paths = dev_test_df['filename'].tolist()
    class_list = np.unique(dev_test_df['scene_label'])
    
else:
    data_val = load_data_2020_evaluate(feat_path, val_csv, num_freq_bin, 'logmel')
    data_deltas_val = deltas(data_val)
    data_deltas_deltas_val = deltas(data_deltas_val)
    data_val = np.concatenate((data_val[:,:,4:-4,:], data_deltas_val[:,:,2:-2,:], data_deltas_deltas_val), axis=-1)
    print(data_val.shape)
    
    dev_test_df = pd.read_csv(val_csv, sep='\t', encoding='ASCII')
    wav_paths = dev_test_df['filename'].tolist()
    
    for idx, elem in enumerate(wav_paths):
        wav_paths[idx] = wav_paths[idx].split('/')[-1]
    
#=========================================================================================================#
interpreter_quant = tf.lite.Interpreter(model_path=model_path)
interpreter_quant.allocate_tensors()

#=========================================================================================================#

if not is_eval:
    overall_acc, preds, preds_class_idx = evaluate_model(interpreter_quant, 
                                                         data_val, 
                                                         y_val, 
                                                         num_class=num_classes,
                                                         is_eval=False)

    over_loss = metrics.log_loss(y_val_onehot, preds)
    print("\nval acc: ", "{0:.4f}".format(overall_acc))
    print("val log loss: ", "{0:.4f}\n".format(over_loss))

    y_pred_val = np.argmax(preds, axis=1)
    conf_matrix = metrics.confusion_matrix(y_val, y_pred_val)
    plot_confusion_matrix(y_val, y_pred_val, class_list, normalize=True, title=None, png_name=csv_path.replace('.csv','.png'))
    
    overall_accuracy = metrics.accuracy_score(y_val, y_pred_val)
    precision_mat = metrics.precision_score(y_val, y_pred_val, average=None, zero_division='warn')
    recall_mat = metrics.recall_score(y_val, y_pred_val, average=None, zero_division='warn')
    f1_score_mat = metrics.f1_score(y_val, y_pred_val, average=None, zero_division='warn')
    precision = metrics.precision_score(y_val, y_pred_val, average='weighted', zero_division='warn')
    recall = metrics.recall_score(y_val, y_pred_val, average='weighted', zero_division='warn')
    f1_score = metrics.f1_score(y_val, y_pred_val, average='weighted', zero_division='warn')

    print(metrics.classification_report(y_val, y_pred_val))
    print(metrics.confusion_matrix(y_val, y_pred_val))

    print("\nper-class precision: ", precision_mat)
    print("\nper-class recall: ", recall_mat)
    print("\nper-class f1-score: ", f1_score_mat)

    print("\naccuracy :", overall_accuracy)
    print("precision :", precision)
    print("recall :", recall)
    print("f1 score :", f1_score)

else:
    preds, preds_class_idx = evaluate_model(interpreter_quant, 
                                            data_val, 
                                            test_labels=None, 
                                            num_class=num_classes,
                                            is_eval=True)
    y_pred_val = np.argmax(preds, axis=1)

#=========================================================================================================#

scene_map_str = """
indoor 0
outdoor 1 
transportation 2
"""

scene_index_map={}
for line in scene_map_str.strip().split('\n'):
    ch, index = line.split()
    scene_index_map[int(index)] = ch
labels = [str(scene_index_map[c]) for c in y_pred_val]
filename = [str(a[:]) for a in wav_paths]
left = {'filename': filename, 'scene_label': labels}
left_df = pd.DataFrame(left)
right_df = pd.DataFrame(preds, columns = ['indoor',
                                          'outdoor',
                                          'transportation'])
merge = pd.concat([left_df, right_df], axis=1, sort=False)
merge.to_csv(csv_path, sep = '\t', index=False)