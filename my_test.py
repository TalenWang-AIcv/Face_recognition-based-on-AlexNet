# -*- use coding:utf-8  -*-
import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 1
train_sets = './Train_data'
tfrecoder_dir = './TFrecoder'
train = os.listdir(os.path.join(train_sets))
train_name = os.path.join(tfrecoder_dir, 'mnist_train.tfrecords')
train_writer = tf.python_io.TFRecordWriter(train_name)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tfrecords():
    for i in range(60000):
        patch, label = mnist.train.next_batch(batch_size, shuffle=False)
        img = np.reshape(patch, [28, 28])
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img * 255
        label = np.argmax(label)
        assert img is not None
        patch_raw = img.astype(np.uint8).tostring()
        label_raw = np.array([label]).astype(np.int32).tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
                                    'patch_raw': _bytes_feature(patch_raw),
                                    'label': _bytes_feature(label_raw)}))
        train_writer.write(example.SerializeToString())


make_tfrecords()
