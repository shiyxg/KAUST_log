# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from time import time
from graph import CNN
from getsamples import DATA


NUM = 5
shape = [256,256]
data = DATA()

pwd = './test_UNET06'


def save_summary(step, sess, trainWrite, g):

    fetchVariables = [g.para[0], g.para[2]]
    trainBatch = data.train_batch(NUM, chose_sample=True)
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

def train():

    g = CNN(shape)
    g.build_graph(NUM)

    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        trainWrite = tf.summary.FileWriter(pwd + '/train', tf.get_default_graph())

        LR = 1e-3
        for i in range(10000):
            trainBatch = data.train_batch(NUM)
            feedData = {g.feed[0]: trainBatch[0], g.feed[1]: trainBatch[1],
                        g.feed[2]: LR, g.feed[3]: True, g.feed[4]: 0.5}
            fetchVariables = g.train
            if np.isnan(trainBatch[0].max()):
                raise ValueError('1')
            sess.run(fetches=fetchVariables, feed_dict=feedData)

            if i % 100 == 0:
                save_summary(i, sess, trainWrite, g)

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


