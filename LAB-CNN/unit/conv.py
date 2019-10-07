# -*- coding = utf-8 -*-
from __future__ import print_function

import tensorflow as tf
import numpy as np


def conv2d(input, conf, trainable=True, core_init=None):
    '''
    to do conv to a value
    :param input: the tensorflow variable before this conv
    :param conf: sample_2d = {
                                'filterSize'        :[3,3],
                                'outputChn'         :256,
                                'strideSize'        :[1,1],
                                'padding'           :'SAME',
                                'filterStddev'      :1
                             }
    :param trainable: if the conv_weight traniable
    :param core_init: conv_init value, for restored parameters from file
    :return: variable after convolution
    '''
    inputChn = input.get_shape().as_list()[-1]
    if len(conf['filterSize']) == 2:
        coreSize_1, coreSize_2 = conf['filterSize']
    elif len(conf['filterSize']) == 1:
        coreSize_1, coreSize_2 = [conf['filterSize'][0], conf['filterSize'][0]]
    else:
        raise ValueError('conf["filterSize"] Error')

    outputChn = conf['outputChn']
    coreShape = [coreSize_1, coreSize_2, inputChn,outputChn]

    # get the stride
    if len(conf['strideSize']) == 2:
        strideSize_1, strideSize_2 = conf['strideSize']
    elif len(conf['strideSize']) == 1:
        strideSize_1, strideSize_2 = [conf['strideSize'][0],conf['strideSize'][0]]
    else:
        strideSize_1, strideSize_2 = [1, 1]

    stride=[1, strideSize_1, strideSize_2, 1]

    if core_init is None:
        if conf.get('filterStddev') is None:
            # Use Glorot and Bengio(2010)'s init method
            n_in = inputChn*coreSize_1*coreSize_2
            n_out = outputChn*coreSize_2*coreSize_1*1.0/(strideSize_1*strideSize_2)
            stddev = np.sqrt(2.0/(n_in+n_out))

            # Gauss disrtribution
            core_init = tf.truncated_normal(mean=0, stddev=stddev, shape=coreShape)
            # uniform distribution
            # core_init = np.random.uniform(low =-np.sqrt(3)*stddev,
            #                               high= np.sqrt(3)*stddev,
            #                               size= coreShape).astype('float32')
        else:
            core_init = np.random.uniform(low=-np.sqrt(3) * conf['filterStddev'],
                                          high=np.sqrt(3) * conf['filterStddev'],
                                          size=coreShape).astype('float32')
    else:
        core_init = core_init

    if conf.get('padding') is not None:
        padding = conf.get('padding')
    else:
        padding = 'SAME'

    with tf.name_scope('conv2d'):
        core = tf.Variable(core_init,name='convCore', trainable=trainable)
        result = tf.nn.conv2d(input, core, strides=stride, padding=padding, data_format='NHWC')
        tf.add_to_collection('conv_core', core)
    return result

