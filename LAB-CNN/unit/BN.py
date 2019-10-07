# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np

BN_EPSILON = 0.001
decay = 0.99


def BN(input, conf, para=None, trainable=False):
    '''
    :param input: data that needs BN
    :param conf: BN's configuration
                conf_sample={'is_training',tf.bool(True), 'only_bias': False}
                only_bias wil only use shift and scale parameters
    :param para: stored data
    :param trainable:
    :return: result after BN
    '''
    if para is None:
        with tf.name_scope('BN'):
            # get the num of chn(conv)/nodes(NN)
            params_shape = input.shape.as_list()[-1]
            # get the shape of input, expect the chn/nodes(the last dimn)
            axis = list(range(len(input.shape.as_list())-1))
            shift = tf.Variable(np.zeros(params_shape).astype('float32'), name='beta')
            scale = tf.Variable(np.ones(params_shape).astype('float32'), name='gamma')

            # moving_mean and moving_variance will store the minibatch's var and mean.
            moving_mean = tf.Variable(np.zeros(params_shape).astype('float32'), trainable=False, name='moving_mean')
            moving_variance = tf.Variable(np.ones(params_shape).astype('float32'), trainable=False, name='moving_mean')
            # minibatch's var and mean
            batch_mean, batch_var = tf.nn.moments(input, axis)

            '''
            assign the mean and var of minibatch to moving ones, thay are operations
            ATTENTION! if train_mean and train_var are not fetched when session run, moving values won't change!!!
            if a value has no connection with the fetched variables, it will not change
            '''
            train_mean = tf.assign(moving_mean, moving_mean*decay + batch_mean*(1-decay))
            train_var = tf.assign(moving_variance, moving_variance*decay + batch_var*(1-decay))

            # based on if we are in training, choose the stored mean or mini-batch_values
            mean, var = tf.cond(conf['is_training'],
                                lambda: (batch_mean, batch_var),
                                lambda: (moving_mean, moving_variance))

            tf.add_to_collection('ops', train_mean)
            tf.add_to_collection('ops', train_var)

            # for debug
            tf.add_to_collection('moving_mean', moving_mean)
            tf.add_to_collection('moving_var', moving_variance)
            tf.add_to_collection('mean', mean)
            tf.add_to_collection('var', var)
            tf.add_to_collection('scale', scale)
            tf.add_to_collection('shift', shift)

            if conf.get('only_bias') is not True:
                # oup = scale*(input-mean)/(var+BN_EPSILON)+shift
                result = tf.nn.batch_normalization(x=input, mean=mean, variance=var, offset=shift, scale=scale,
                                               variance_epsilon=BN_EPSILON)
            else:
                result = input*scale+shift

    # if we want to use values in file
    elif para is not None:
        with tf.name_scope('BN'):
            x = input
            params_shape = x.shape.as_list()[-1]
            # make the stored para to BN
            axis = list(range(len(x.shape.as_list()) - 1))  # 得到需要计算batch的部分，除了最后一个维度不进行
            shift = tf.Variable(para[0, :].astype('float32'), name='beta', trainable=trainable)
            scale = tf.Variable(para[1, :].astype('float32'), name='gamma', trainable=trainable)
            moving_mean = tf.Variable(para[2, :], trainable=False, name='moving_mean')
            moving_variance = tf.Variable(para[3, :], trainable=False, name='moving_mean')
            batch_mean, batch_var = tf.nn.moments(x, axis)

            train_mean = tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(moving_variance, moving_variance * decay + batch_var * (1 - decay))
            mean, var = tf.cond(conf['is_training'],
                                lambda: (batch_mean, batch_var),
                                lambda: (moving_mean, moving_variance))

            tf.add_to_collection('moving_mean', moving_mean)
            tf.add_to_collection('moving_var', moving_variance)
            tf.add_to_collection('mean', mean)
            tf.add_to_collection('var', var)
            tf.add_to_collection('ops', train_mean)
            tf.add_to_collection('ops', train_var)
            tf.add_to_collection('scale', scale)
            tf.add_to_collection('shift', shift)

            if conf.get('only_bias') is not True:
                result = tf.nn.batch_normalization(x=input, mean=mean, variance=var, offset=shift, scale=scale,
                                                   variance_epsilon=BN_EPSILON)
            else:
                result = input * scale + shift

    return result

