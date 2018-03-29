# -*- use coding:utf-8  -*-
import tensorflow as tf
import numpy as np


def maxPoolLayer(X, kHeight, kWidth, strideX, strideY, name, padding='SAME'):
    return tf.nn.max_pool(X, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1],
                          padding=padding, name=name)


def dropout(X, keepPro, name=None):
    return tf.nn.dropout(X, keep_prob=keepPro, name=name)


def LRN(X, R, alpha, beta, name=None, bias=1.0):
    return tf.nn.local_response_normalization(X, depth_radius=R, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def fcLayer(X, inputD, outputD, reluFlag, name):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('W', shape=[inputD, outputD], dtype='float', initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [outputD], dtype='float', initializer=tf.zeros_initializer())
        out = tf.nn.xw_plus_b(X, W, b, name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out


def convLayer(X, kHeight, kWidth, strideX, strideY, featureNum, name, padding='SAME', group=1):
    channel = int(X.get_shape()[-1])
    conv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, strideX, strideY, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('W', shape=[kHeight, kWidth, channel/group, featureNum], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[featureNum], initializer=tf.zeros_initializer())

        xNew = tf.split(value=X, num_or_size_splits=group, axis=3)
        wNew = tf.split(value=W, num_or_size_splits=group, axis=3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(axis=3, values=featureMap)
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name=scope.name)


class AlexNet(object):
    def __init__(self, X, keepPro, classNum, skip, modelPath='bvlc_alexnet.npy'):
        self.X = X
        self.KeepPro = keepPro
        self.ClassNum = classNum
        self.Skip = skip
        self.ModelPath = modelPath
        self.buildCNN()

    def buildCNN(self):
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, name='conv1', padding='VALID')
        lrn1 = LRN(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, name='pool1', padding='VALID')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, name='conv2', group=2)
        lrn2 = LRN(conv2, 2, 2e-05, 0.75, name='lrn2')
        pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, name='pool2', padding='VALID')

        # 3rd Layer: Conv (w ReLu)
        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, name='conv3')
        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, name='conv4', group=2)
        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, name='conv5', group=2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, name='pool5', padding='SAME')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        fcIn = tf.reshape(pool5, [-1, 256*6*6])
        fc1 = fcLayer(fcIn, 256*6*6, 4096, True, name='fc6')
        dropout1 = dropout(fc1, self.KeepPro)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc2 = fcLayer(dropout1, 4096, 4096, True, name='fc7')
        dropout2 = dropout(fc2, self.KeepPro)

        # 8th Layer: FC and return unscaled activations
        self.fc3 = fcLayer(dropout2, 4096, self.ClassNum, True, name='fc8')

    def loadModel(self, sess):
        # Load the weights into memory
        wDict = np.load(self.ModelPath, encoding='bytes').item()
        # Loop over all layer names stored in the weights dict
        for name in wDict:
            if name == u'fc8':
                continue
            # Check if layer should be trained from scratch
            if name not in self.Skip:
                with tf.variable_scope(name, reuse=True):
                    # Assign weights/biases to their corresponding tf variable
                    for data in wDict[name]:
                        if len(data.shape) == 1:
                            sess.run(tf.get_variable('b', trainable=False).assign(data))
                        else:
                            sess.run(tf.get_variable('W', trainable=False).assign(data))

