import numpy
import config
import table
import os


PATH = 'feat/'

NAME = ['baby']

result_file_path = 'ground_truth/example'

with open(result_file_path, 'w') as f:
    for class_name in NAME:
        file_list = os.listdir(PATH)
        for file_name in file_list:
            f.write(file_name+'\t'+class_name)
            f.write('\n')

