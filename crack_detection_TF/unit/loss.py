import tensorflow as tf


def weight_cross_entropy(label, logits, pos=None):
    '''
    带权重的交叉熵作为评估函数
    :param label: 理想结果
    :param logits: 模型运算得到的结果
    :return: 返回平均的交叉熵
    '''
    with tf.name_scope('Loss'):
        if pos is None:
            numNF = tf.reduce_sum(1 - label)
            numF = tf.reduce_sum(label)
            posWeight = numNF * 1.0 / numF
            cost = tf.nn.weighted_cross_entropy_with_logits(label, logits, posWeight)
            result = tf.reduce_sum(cost * numF * 1.0 / (numF + numNF))
        else:
            cost = tf.nn.weighted_cross_entropy_with_logits(label, logits, pos)
            result = tf.reduce_sum(cost * 1 * 1.0 / (pos+1))
    return result


def sigmoid_cross_entropy(label, logits):
    with tf.name_scope('Loss'):
        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
        cost = tf.reduce_mean(cost)
    return cost


def norm2(label, logits):
    with tf.name_scope('Loss'):
        result = tf.reduce_sum((label-logits)**2)
    return result
