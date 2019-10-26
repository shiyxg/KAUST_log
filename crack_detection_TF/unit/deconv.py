# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
from unit.BN import BN


def bilinear(size, scale):
    '''
    这是为了形成一个双线性矩阵的初始化，只对二维的反卷积成立，而且建议使用BN
    '''
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


def deconv3d(input, conf, use_bilinear=False, trainable=True):
    '''
    still some problems when deal with 2d
    :param input:
    :param conf:
    :param use_bilinear:
    :return:
    '''
    input_size = input.shape.as_list()
    input_chn = input_size[4]

    scale = conf['scale']
    stride = [1, scale, scale, scale, 1]
    output_chn = conf['outputChn']
    filter_size = 2 * scale - scale % 2
    deconv_core_size = [filter_size, filter_size, filter_size, output_chn, input_chn]

    if conf.get('outputSize') is None:
        output_shape = [tf.shape(input)[0],
                        tf.shape(input)[1]*scale,
                        tf.shape(input)[2]*scale,
                        tf.shape(input)[3]*scale,
                        output_chn]
    else:
        output_shape = [tf.shape(input)[0],
                        conf['outputSize'][0],
                        conf['outputSize'][1],
                        conf['outputSize'][2],
                        output_chn]

    if use_bilinear:
        deconv_init = bilinear(deconv_core_size, scale)
    else:
        n_in = input_chn * 2 * 2
        n_out = output_chn * 2 * 2 * 1.0 / (1 * 1)
        stddev = np.sqrt(2) / np.sqrt(n_in + n_out)
        deconv_init = np.random.uniform(low=-np.sqrt(3) / stddev,
                                        high=np.sqrt(3) / stddev,
                                        size=deconv_core_size).astype('float32')

    with tf.name_scope('deConvolution'):
        core = tf.Variable(deconv_init, name='deconvCore', trainable=trainable)
        deconv_result = tf.nn.conv2d_transpose(input, core, output_shape, strides=stride, padding='SAME')
        tf.add_to_collection('deconv_cores', core)

    return deconv_result


def deconv2d_layer(input, conf):
    '''
    这是构建一个完整的反卷积层的函数
    :param input: 反卷积层的输入，要求格式NHWC
    :param conf: 反卷积层有关的参数，包括：
                层名字：name
                二维反卷积要用到的：scale， outputChn， outputShape, 可选的：strideSize
                BN操作要用到的：is_training
    :return:经过卷积层之后的值
    '''

    with tf.name_scope(conf['name']):
        result = deconv2d(input, conf, use_bilinear=True)
        result = BN(result, conf)
        result = tf.nn.relu(result)
    return result


def deconv1d(input, conf):
    input_size = input.shape.as_list()
    input_chn = input_size[2]
    input_4d = tf.reshape(input, [-1, 1, input_size[1], input_size[2]])

    scale = conf['scale']
    stride = [1, 1, scale, 1]
    output_chn = conf['outputChn']
    filter_size = 2 * scale - scale % 2
    deconv_core_size = [1, filter_size, output_chn, input_chn]

    if conf.get('outputSize') is None:
        output_shape = [tf.shape(input)[0],
                        1,
                        tf.shape(input)[2] * scale,
                        output_chn]
    else:
        output_shape = [tf.shape(input)[0],
                        1,
                        conf['outputSize'][0],
                        output_chn]

    n_in = input_chn * 2
    n_out = output_chn * 2 * scale * 1
    stddev = np.sqrt(2) / np.sqrt(n_in + n_out)
    deconv_init = np.random.uniform(low=-np.sqrt(3) / stddev,
                                    high=np.sqrt(3) / stddev,
                                    size=deconv_core_size).astype('float32')

    with tf.name_scope('deConvolution'):
        core = tf.Variable(deconv_init, name='deconvCore')
        deconv_result = tf.nn.conv2d_transpose(input_4d, core, output_shape, strides=stride, padding='SAME')

    deconv_result_3d = tf.reshape(deconv_result, [-1, input_size[1], input_size[2]])

    return deconv_result_3d


def deconv1d_layer(input, conf):
    '''
    这是构建一个完整的反卷积层的函数
    :param input: 反卷积层的输入，要求格式NHWC
    :param conf: 反卷积层有关的参数，包括：
                层名字：name
                二维反卷积要用到的：scale， outputChn， outputShape, 可选的：strideSize
                BN操作要用到的：is_training
    :return:经过卷积层之后的值
    '''

    with tf.name_scope(conf['name']):
        result = deconv1d(input, conf)
        result = BN(result, conf)
        result = tf.nn.relu(result)
    return result


conf_sample_2d = {
    'name'              :'deconv',
    'scale'             :2,
    'outputChn'         :128,
    'outputSize'       :[128,128],
    'is_training'       :True
}

conf_sample_1d = {
    'name'              :'deconv',
    'scale'             :2,
    'outputChn'         :128,
    'outputSize'       :[128],
    'is_training'       :True
}
