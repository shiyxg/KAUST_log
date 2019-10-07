import tensorflow as tf


def max_pool(input, name='pool', conf={}):

    if conf.get('ksize') is None:
        ksize=[1, 2, 2, 1]
    else:
        ksize=conf['ksize']

    if conf.get('stride') is None:
        stride=[1, 2, 2, 1]
    else:
        stride=conf['stride']

    with tf.name_scope(name):
        output = tf.nn.max_pool(input, ksize=ksize, strides=stride, padding='SAME')
    return output


def max_pool3d(input, name='pool', conf={}):

    if conf.get('ksize') is None:
        ksize=[1, 2, 2, 2, 1]
    else:
        ksize=conf['ksize']

    if conf.get('stride') is None:
        stride=[1, 2, 2, 2, 1]
    else:
        stride=conf['stride']

    with tf.name_scope(name):
        output = tf.nn.max_pool3d(input, ksize=ksize, strides=stride, padding='SAME')
    return output


def max_pool_argmax(input, conf={}):
    if conf.get('ksize') is None:
        ksize=[1, 2, 2, 1]
    else:
        ksize=conf['ksize']

    if conf.get('stride') is None:
        stride = ksize
    else:
        stride=conf['stride']

    if conf.get('name') is None:
        name = 'pool'
    else:
        name = conf['name']

    with tf.name_scope(name):
        with tf.name_scope('get_indices'):
            output, indices = tf.nn.max_pool_with_argmax(input, ksize=ksize, strides=stride, padding='SAME')
            indices = tf.stop_gradient(indices)
    return [output, indices]


# thanks https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/WhatWhereAutoencoder.py
def max_unpool_2d(input, indices, conf={}):
    if conf.get('output_shape') is None:
        raise ValueError('do not set output_shape, it could make some mistakes')
    else:
        output_shape = conf['output_shape']
        output_size = output_shape[0] * output_shape[1]

    if conf.get('name') is None:
        name = 'unpool'
    else:
        name = conf['name']

    with tf.name_scope(name):
        batch_num = input.get_shape().as_list()[0]
        chn_num = input.get_shape().as_list()[-1]

        input_1d = tf.reshape(input, [tf.size(input)])
        indices_1d = tf.reshape(indices, [tf.size(input), 1])

        shape = tf.constant([batch_num*output_size*chn_num])
        shape = tf.cast(shape, tf.int64)

        result = tf.scatter_nd(indices_1d, input_1d, shape)
        # print(shape)
        result = tf.reshape(result, [batch_num, output_shape[0], output_shape[1], chn_num])

    return result


def max_pool_1d(input, conf={}):

    if conf.get('ksize') is None:
        ksize=[4]
    else:
        ksize = [conf['ksize'][1]]

    if conf.get('stride') is None:
        stride=ksize
    else:
        stride=[conf['stride'][1]]

    with tf.name_scope(conf['name']):
        output = tf.nn.pool(input, window_shape=ksize, pooling_type='MAX', strides=stride, padding='SAME')
    return output