# -*- use coding:utf-8  -*-
import tensorflow as tf
from random import shuffle
import numpy as np
import cv2
import os

imgs = []
labs = []
train_sets = './Train_data'
tfrecoder_dir = './TFrecoder'
train = os.listdir(os.path.join(train_sets))
train_name = os.path.join(tfrecoder_dir, 'face224_train.tfrecords')
train_writer = tf.python_io.TFRecordWriter(train_name)

label_map = {'wang': 0, 'train001': 1, 'train002': 2, 'train003': 3,
             'train004': 4, 'train005': 5, 'train006': 6, 'train007': 7,
             'train008': 8, 'train009': 9, 'train010': 10, 'train011': 11,
             'train012': 12, 'train013': 13, 'train014': 14, 'train015': 15,
             'train016': 16, 'train017': 17, 'train018': 18, 'train019': 19,
             'train020': 20, 'train021': 21, 'train022': 22, 'train023': 23,
             'train024': 24, 'train025': 25, 'train026': 26, 'train027': 27,
             'train028': 28, 'train029': 29, 'train030': 30, 'train031': 31,
             'train032': 32, 'train033': 33, 'train034': 34, 'train035': 35,
             'train036': 36, 'train037': 37, 'train038': 38, 'train039': 39,
             'train040': 40, 'train041': 41, 'train042': 42, 'train043': 43,
             'train044': 44, 'train045': 45, 'train046': 46, 'train047': 47,
             'train048': 48, 'train049': 49, 'train050': 50, 'train051': 51}


# 读取数据集
def readData(path):
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        for images in os.listdir(child_path):
            if images.endswith('.jpg'):
                img = cv2.imread(os.path.join(child_path, images))
                imgs.append(img)
                labs.append(label_map[child_dir])
    return imgs, labs


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tfrecords():
    patch, label = readData(train_sets)
    idx = range(len(patch))
    shuffle(idx)

    for i in idx:
        patch_raw = patch[i].tostring()
        label_raw = np.array([label[i]]).astype(np.int32).tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
                                    'patch_raw': _bytes_feature(patch_raw),
                                    'label': _bytes_feature(label_raw)}))
        train_writer.write(example.SerializeToString())


make_tfrecords()
print 'Make Tfrecords finished!!!'
