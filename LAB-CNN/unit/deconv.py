# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np


def bilinear(size, scale):
    init = np.zeros(size)
    assert size[0] == size[1]
    filterSize = size[0]
    filter = np.zeros([filterSize, filterSize])
    if filterSize % 2 == 0:
        center = scale - 0.5
    elif filterSize % 2 == 1:
        center = scale - 1
    else:
        raise ValueError('filter Size is not set')

    for i in range(filterSize):
        for j in range(filterSize):
            filter[i, j] = (1 - abs(i - center) * 1.0 / scale) * (1 - abs(j - center) * 1.0 / scale)

    for i in range(size[2]):
        for j in range(size[3]):
            init[:, :, i, j] = filter/size[3]/scale/scale

    return init.astype('float32')


def deconv2d(input, conf, use_bilinear=False, trainable=True, core_init=None):
    '''
    still some problems when deal with 2d
    :param input:
    :param conf:
    :param use_bilinear:
    :return:
    '''
    input_size = input.shape.as_list()
    input_chn = input_size[3]

    scale = conf['scale']
    stride = [1, scale, scale, 1]
    output_chn = conf['outputChn']
    filter_size = 2 * scale - scale % 2
    deconv_core_size = [filter_size, filter_size, output_chn, input_chn]

    if conf.get('outputSize') is None:
        output_shape = [tf.shape(input)[0],
                        tf.shape(input)[1]*scale,
                        tf.shape(input)[2]*scale,
                        output_chn]
    else:
        output_shape = [tf.shape(input)[0],
                        conf['outputSize'][0],
                        conf['outputSize'][1],
                        output_chn]

    if core_init is None:
        if use_bilinear:
            deconv_init = bilinear(deconv_core_size, scale)
        else:
            n_in = input_chn * 2 * 2
            n_out = output_chn * 2 * 2 * 1.0 / (1 * 1)
            stddev = np.sqrt(2) / np.sqrt(n_in + n_out)
            deconv_init = np.random.uniform(low=-np.sqrt(3) / stddev,
                                            high=np.sqrt(3) / stddev,
                                            size=deconv_core_size).astype('float32')
    else:
        deconv_init = core_init

    with tf.name_scope('deConvolution'):
        core = tf.Variable(deconv_init, name='deconvCore', trainable=trainable)
        deconv_result = tf.nn.conv2d_transpose(input, core, output_shape, strides=stride, padding='SAME')
        tf.add_to_collection('deconv_cores', core)

    return deconv_result
