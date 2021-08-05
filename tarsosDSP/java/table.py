import sys
import os
import numpy as np

import utils


class ClassTable:
    # nClass = 0
    # classList = None

    def __init__(self, class_list):
        self.nClass = len(class_list)
        self.classList = class_list

    def class2id(self, class_str):
        try:
            idx = self.classList.index(class_str)
        except ValueError:
            print(ValueError)
            idx = -1
        return idx

    def class2id_numpy(self, class_list):
        ids = np.zeros(len(class_list))
        for idx, val in enumerate(class_list):
            ids[idx] = self.class2id(val)
        return ids

    def id2class(self, idx):
        try:
            class_str = self.classList[idx]
        except IndexError:
            print(IndexError)
            class_str = None
        return class_str


class GroundTruthTable(ClassTable):
    ground_truth = dict()

    def __init__(self, class_list, meta_file):
        self.nClass = len(class_list)
        self.classList = class_list
        with open(meta_file, 'r') as f:
            for ln in f.readlines():
                [filename, answer] = [ ln.strip().split('\t')[0], ln.strip().split('\t')[1] ]
                key = utils.gen_key_from_file(filename)
                self.ground_truth[key] = (answer, self.class2id(answer))
        return

    def key2id(self, key):
        try:
            id = self.ground_truth[key][1]
        except:
            id = -1
        return id

    def key2class(self, key):
        try:
            classstr = self.ground_truth[key][0]
        except:
            classstr = None
        return classstr

    def read_bbcsoundfx(self, bbcfile):
        with open(bbcfile, 'r') as f:
            for ln in f.readlines():
                [filename, answer_idx] = [ ln.strip().split(' ')[0], ln.strip().split(' ')[1] ]
                key = utils.gen_key_from_file(filename)
                self.ground_truth[key] = (self.id2class(int(answer_idx)), answer_idx)

    def read_freesound(self, freesoundfile):
        with open(freesoundfile, 'r') as f:
            for ln in f.readlines():
                lnstrip = ln.strip().split('_')
                if len(lnstrip) == 4:
                    classname = lnstrip[0]+'_'+lnstrip[1]
                else:
                    classname = lnstrip[0]

                key = utils.gen_key_from_file(ln)
                self.ground_truth[key] = (classname, self.class2id(classname))

    def read_urbansound8k(self, urban8kfile):
        with open(urban8kfile, 'r') as f:
            for ln in f.readlines():
                [key, classname] = [ ln.strip().split(' ')[0], ln.strip().split(' ')[1] ]
                self.ground_truth[key] = (classname, self.class2id(classname))