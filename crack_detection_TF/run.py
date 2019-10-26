# -*- coding: utf-8 -*-

from __future__ import print_function

# Graph is built in Graph.py
import tensorflow as tf
import sys
import numpy as np
import sys
import scipy.io as io
import os
import matplotlib as mpl
from pylab import *
from time import time
from graph import *
from get_samples import DATA, DATA_BIG, DATA_gee
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

NUM = 5
shape = [256,256]
# data = DATA()
data = DATA_BIG(num=8, from_file=False, downsample=2)
# data = DATA_gee(num=8, from_file=True, downsample=2, file = 'C:/Users/shiyx/Documents/ML_log/crack_detection/VGG/test_VGG03/all.npy')
data.dataset(3000)
pwd = 'C:/Users/shiyx/Documents/ML_log/crack_detection/VGG/test_UNET06'
# assert  1==2

def save_summary_fcn(step, sess, trainWrite, g):

    fetchVariables = [g.para[0], g.para[2]]
    trainBatch = data.trainBatch(NUM, chose_sample=True)
    # trainBatch = data.trainBatch(NUM, 0.5, shape)
    feedData_train = {g.feed[0]: trainBatch[0], g.feed[1]: trainBatch[1],
                      g.feed[2]: 1e-3, g.feed[3]: False, g.feed[4]: 1}

    [loss_train, summary_train
     ] = sess.run(fetches=fetchVariables, feed_dict=feedData_train)
    trainWrite.add_summary(summary_train, step)
    trainWrite.flush()

    if step%500==0:
        g.save_para_fcn(sess, pwd + '/para_npy/%s.npy' % (step + 1), NUM)
    print('*********update log***************')
    print('step:%06d' % step, 'train_loss:%0.4f'%loss_train)

def train_fcn():
    g = FCN_UNET(shape)
    g.output_chn = 1
    g.input_chn = data.trainBatch(1)[0].shape[3]
    g.build_graph(NUM)

    with tf.Session(graph=g.graph) as sess:


        ops = tf.get_collection('ops')
        sess.run(tf.global_variables_initializer())
        trainWrite = tf.summary.FileWriter(pwd + '/train', tf.get_default_graph())

        LR = 1e-3
        for i in range(10000):
            trainBatch = data.trainBatch(NUM)
            feedData = {g.feed[0]: trainBatch[0], g.feed[1]: trainBatch[1],
                        g.feed[2]: LR, g.feed[3]: True, g.feed[4]: 0.5}
            fetchVariables = [g.train, ops, g.para[0], g.para[1]]
            if np.isnan(trainBatch[0].max()):
                raise ValueError('1')
            [_, _, train_loss, result] = sess.run(fetches=fetchVariables, feed_dict=feedData)
            # print(result.max())
            # print(result.min())

            if i % 100 == 0:
                save_summary_fcn(i, sess, trainWrite, g)
            if i<1000:
                LR=1e-3
            elif i<3000:
                LR=5e-4
            elif i<5000:
                LR=2e-4
            else:
                LR=1e-4
        trainWrite.close()

    return sess


train_fcn()

