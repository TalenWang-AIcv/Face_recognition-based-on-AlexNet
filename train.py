# -*- use coding:utf-8  -*-
import tensorflow as tf
import os

from AlexNet_CNN import AlexNet
from input_pipeline import input_pipeline

ckpt_dir = './ckpt'
batch_size = 100
img_size = 224
# 后3个参数分别是batch_size, img_size, read_thread
images, labels = input_pipeline(
        [os.path.join('/home/talentwong/Graduate/Face/TFrecoder', 'face224_train.tfrecords')], batch_size, img_size, 1)

skip = []
classNum = 52                                                        # 52类
# define placeholder for inputs to network
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [batch_size, img_size, img_size, 3], name='x-input')  # 3 -> 1
    y = tf.placeholder(tf.int32, [None, 1], name='y-input')                           # 1 -> 10
keep_pro = tf.placeholder(tf.float32)
model = AlexNet(X=x, keepPro=keep_pro, classNum=classNum, skip=skip)

prediction = model.fc3
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(tf.reshape(y, [-1]), tf.int32), logits=prediction)  # loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)
# minimize loss' train_step
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

saver = tf.train.Saver(name="saver")

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    if tf.gfile.Exists(os.path.join(ckpt_dir, 'checkpoint')):
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
    else:
        sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # writer
    writer = tf.summary.FileWriter("logs", sess.graph)
    # merged
    merged = tf.summary.merge_all()

    model.loadModel(sess)
    image, label = sess.run([images, labels])
    for i in range(10001):
        _, c = sess.run([train_step, loss], feed_dict={x: image, y: label, keep_pro: 0.5})
        if i % 20 == 0:
            image, label = sess.run([images, labels])
            result = sess.run(merged, feed_dict={x: image, y: label, keep_pro: 1.})
            writer.add_summary(result, i)
            saver.save(sess, ckpt_dir + os.sep + 'Alexnet.ckpt')
            print i, 'loss:', c
    coord.request_stop()
    coord.join(threads)

