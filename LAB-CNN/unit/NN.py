import tensorflow as tf
import numpy as np


def NN(x, conf, trainable=True, core_init=None):
    input_length = x.shape.as_list()[1]
    output_length = conf['num']

    if core_init is None:
        weight_size = [input_length, output_length]

        # Use Glorot and Bengio(2010)'s init method
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
        result = tf.nn.xw_plus_b(x=x, weights=w, biases=b)

        tf.add_to_collection('w', w)
        tf.add_to_collection('b', b)

    return result

conf_sample = {
    'num': 1024
}