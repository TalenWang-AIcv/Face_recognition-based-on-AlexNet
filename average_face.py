# -*- use coding:utf-8  -*-

import tensorflow as tf
import numpy as np
import cv2
import os

from input_pipeline import input_pipeline


batch_size = 100
img_size = 224
# 后3个参数分别是batch_size, img_size, read_thread
images, labels = input_pipeline(tf.train.match_filenames_once(
        os.path.join('./TFrecoder', '*.tfrecords')), batch_size, img_size, 1)

sum_r = np.mat(np.zeros((img_size, img_size)))
sum_g = np.mat(np.zeros((img_size, img_size)))
sum_b = np.mat(np.zeros((img_size, img_size)))
average_r = np.mat(np.zeros((img_size, img_size)))
average_g = np.mat(np.zeros((img_size, img_size)))
average_b = np.mat(np.zeros((img_size, img_size)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(100):
        image, label = sess.run([images, labels])
        for n in range(len(image)):
            sum_r += np.array(image[n, :, :, 2])
            sum_g += np.array(image[n, :, :, 1])
            sum_b += np.array(image[n, :, :, 0])
    average_b = sum_b * 1e-4
    average_g = sum_g * 1e-4
    average_r = sum_r * 1e-4

    # while True:
    #     cv2.imshow('b', average_b)
    #     cv2.imshow('g', average_g)
    #     cv2.imshow('r', average_r)
    #     if cv2.waitKey(10) & 0xff is 27:
    #         break
    # cv2.destroyAllWindows()

    coord.request_stop()
    coord.join(threads)



