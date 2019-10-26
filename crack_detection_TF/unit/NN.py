import tensorflow as tf
import numpy as np
from unit.BN import BN

def nn_layer(input, conf, act = tf.nn.relu, keep_prob=None):
    '''
    :param input: the input para, need shape:[batch, length]
    :param conf: about this layer's neuron number
    :return: result after a act(w*x+b) action
    建立一个完整的NN层
    '''
    input_length = input.shape.as_list()[1]
    output_length = conf['num']
    weight_size = [input_length, output_length]

    # # Use Glorot and Bengio(2010)'s init method
    stddev = np.sqrt(2) / np.sqrt(input_length + output_length)

    weight_init = np.random.uniform(low=-np.sqrt(3)*stddev,
                                    high=np.sqrt(3)*stddev,
                                    size=weight_size).astype('float32')

    with tf.name_scope(conf['name']):
        w = tf.Variable(weight_init, name='weight')
        b = tf.Variable(np.zeros(output_length).astype('float32'), name='bias')
        result = input
        result = tf.nn.xw_plus_b(x=result, weights=w, biases=b)
        tf.add_to_collection('wx_result', result)
        result = BN(result, conf)
        if keep_prob is not None:
            result = tf.nn.dropout(result, keep_prob=keep_prob)
        tf.add_to_collection('BN_result', result)
        result = act(result, name='act')
    return result


def NN(input, conf, trainable=True, core_init=None, keep_prob=None):
    input_length = input.shape.as_list()[1]
    output_length = conf['num']
    if core_init is None:
        weight_size = [input_length, output_length]

        # # Use Glorot and Bengio(2010)'s init method
        stddev = np.sqrt(2) / np.sqrt(input_length + output_length)

        weight_init = np.random.uniform(low=-np.sqrt(3)*stddev,
                                        high=np.sqrt(3)*stddev,
                                        size=weight_size).astype('float32')
        bias_init = np.zeros(output_length).astype('float32')
    else:
        weight_init = core_init[0]
        bias_init =core_init[1]

    with tf.name_scope('wx_p_b'):
        w = tf.Variable(weight_init, name='weight', trainable=trainable)
        b = tf.Variable(bias_init, name='bias', trainable=trainable)
        result = input
        result = tf.nn.xw_plus_b(x=result, weights=w, biases=b)
        tf.add_to_collection('w', w)
        tf.add_to_collection('b', b)
        if keep_prob is not None:
            result = tf.nn.dropout(result, keep_prob=keep_prob)
    return result

conf_sample = {
    'name': 'NNL',
    'num': 1024
}