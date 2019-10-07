import tensorflow as tf
from unit.conv import conv2d
from unit.pool import max_pool_argmax
from unit.BN import BN
from unit.NN import NN
import os
import numpy as np

class CNN(object):
    def __init__(self, input_shape, outputclass_num):
        self.graph = tf.Graph() # graph
        self.feed = [] # feed parameters
        self.train = None # train_ops parameters lists
        self.INPUT_SHAPE = input_shape # input size
        self.OUTPUT_SHAPE = [outputclass_num] # output size
        self.class_num = outputclass_num
        self.para = [] # some parameter needed fetch from graph when train
        self.loss = None # loss
        self.raw_oup = None

        # store the conv layers' result
        self.conv_result = []

        # store for networks Variables
        self.conv = []
        self.bn = []
        self.pool = []
        self.nn = []

        # used to stored para as a .npy file and read them from a npy file
        self.conv_init = {}
        self.nn_init={}
        self.bn_init = {}
        self.stru = []
        self.bn_i = 0
        self.conv_i = 0
        self.deconv_i = 0
        self.nn_i = 0

        # the class num of CNN's output
        self.output_nodes = None
        self.conv_trainable = True

    def cnn_network(self):
        [images, _, _, is_training, keep] = self.feed

        # configuration for BN
        conf1 = {'is_training': is_training, 'only_bias': True}
        # configuration for conv
        conf2 = {'filterSize': None, 'outputChn': None}
        # for NN
        conf3 = {'num': 1024}

        # convolution layer 1
        with tf.name_scope('conv1'):
            conf2['filterSize'] = [3, 3]
            conf2['outputChn'] = 32
            x = self.CONV(images, conf2)
            x = self.BN(x, conf1)
            x = self.activate(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool1', 'ksize': [1,2,2,1]})

        # convolution layer 2
        with tf.name_scope('conv2'):
            conf2['filterSize'] = [3, 3]
            conf2['outputChn'] = 64
            x = self.CONV(x, conf2)
            x = self.BN(x, conf1)
            x = self.activate(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool2', 'ksize': [1, 2, 2, 1]})

        # convolution layer 3
        with tf.name_scope('conv3'):
            conf2['filterSize'] = [3, 3]
            conf2['outputChn'] = 128
            x = self.CONV(x, conf2)
            x = self.BN(x, conf1)
            x = self.activate(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool3', 'ksize': [1, 2, 2, 1]})

        # convolution layer 4
        with tf.name_scope('conv4'):
            conf2['filterSize'] = [3, 3]
            conf2['outputChn'] = 256
            x = self.CONV(x, conf2)
            x = self.BN(x, conf1)
            x = self.activate(x)
        self.conv_result.append(x)

        x = self.flatten(x)

        # NN1
        with tf.name_scope('NN1'):
            conf3['num']=1024
            x = self.NN(x, conf3)
            x = self.BN(x, conf2)
            x = self.activate(x)
            x = self.dropout(x, keep_probability=keep)
        # NN2
        with tf.name_scope('NN2'):
            conf3['num']=self.class_num
            x = self.NN(x, conf3)
            x = self.activate(x)
            x = tf.nn.softmax(x)

        return x

    def build_graph(self):
        '''
        this function will be used to build a CNN network with your graph structure
        :param NUM: the number of samples in a mini-batch
        :return: graph with cnn structure
        '''
        with self.graph.as_default():
            # placeholder
            with tf.name_scope('ImagesLabels'):
                images = tf.placeholder(tf.float32, shape=[None]+self.INPUT_SHAPE)
                labels = tf.placeholder(tf.float32, shape=[None]+self.OUTPUT_SHAPE)
            learning_rate = tf.placeholder(tf.float32, name='LR')
            keep = tf.placeholder(tf.float32, name='keep') # for dropout's keep rate
            is_training = tf.placeholder(tf.bool, name='BN_controller') # this placeholder is used for BN

            # placeholder must have some values to feed, so we store them in self.feed
            self.feed = [images, labels, learning_rate, is_training, keep]

            x = self.cnn_network()
            loss = self.loss_define(x, labels)
            # train and others
            with tf.name_scope('other'):
                train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
                result = tf.argmax(x, axis=1)
                self.raw_oup = x

            summary = self.summary(loss, result)

            ops = tf.get_collection('ops')
            self.train = [train,ops]
            self.para = [loss, result, summary]

        return self.graph

    def loss_define(self, x, labels):
        with tf.name_scope('WightCrossEntro'):
            cost = labels * tf.log(x) + (1-labels)*tf.log(1-x)
            loss = tf.reduce_mean(cost)
            self.loss = loss
        return loss

    def summary(self, loss, result):
        # get collection of conv_weight, nn_weight, BN et.al
        self.collection()

        with tf.name_scope('summary'):
            images = self.feed[0]
            labels = self.feed[1]
            is_equal = tf.equal(tf.argmax(labels, axis=1), result)
            accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))


            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.image('input', self.feed[0])



            for i in range(len(self.conv)):
                tf.summary.histogram('conv%sDist' % i, self.conv[i])
            for i in range(len(self.nn)):
                tf.summary.histogram('nn%sDist' % i, self.nn[i])
            summary = tf.summary.merge_all()

        return summary

    def collection(self):

        self.conv = tf.get_collection('conv_core')
        for i in range(len(tf.get_collection('shift'))):
            self.bn.append(
                [tf.get_collection('shift')[i], tf.get_collection('scale')[i], tf.get_collection('moving_mean')[i],
                 tf.get_collection('moving_var')[i]])
        for i in range(len(tf.get_collection('w'))):
            self.nn.append(
                [tf.get_collection('w')[i], tf.get_collection('b')[i]]
            )

    def save_para_fcn(self, sess, file_name, NUM=1):

        c = file_name.split('/')
        c.pop(-1)
        path = ''
        for i in c:
            path = path + '/' + i
        path = path[1:]
        if not os.path.exists(path):
            os.mkdir(path)

        feedData = {self.feed[0]: np.zeros([NUM]+self.INPUT_SHAPE).astype('float32'),
                    self.feed[1]: np.zeros([NUM]+self.OUTPUT_SHAPE).astype('float32'),
                    self.feed[2]: 1e-3, self.feed[3]: False, self.feed[4]: 1}
        fetchVariables = [self.conv, self.bn, self.nn]
        # get weights
        [cores, bn, nn] = sess.run(fetches=fetchVariables, feed_dict=feedData)

        # save them
        print('*******restoreing para*********')
        dic = {'conv_init': {}, 'bn_init': {}, 'nn_init':{}}
        for i in range(len(cores)):
            dic['conv_init'][i] = np.array(cores[i])
        for i in range(len(bn)):
            dic['bn_init'][i] = np.array(bn[i])
        for i in range(len(nn)):
            dic['nn_init'][i] = np.array(nn[i])
        dic['structure'] = self.stru
        np.save(file_name, dic)

        print('stored bn as npy, :finished')

    def restore_para(self, filename, FROMCNN=False):
        '''
        这是一个启用网络参数的函数，主要包括bn,卷积核与反卷积核，以及用于重建网络的一个字符串
        :param filename:
        :return:
        '''
        para_dict = np.load(filename).tolist()

        self.bn_init = para_dict['bn_init']
        self.conv_init = para_dict['conv_init']
        self.nn_init = para_dict['nn_init']

    def flatten(self, x):
        # reshape x to 2d tensor
        with tf.name_scope('flatten'):
            _, w, h, c = tf.shape(x)
            x = tf.reshape(x, [-1,w,h,c])# make x a 2d tensor rather than 4d tensor
        return x

    def activate(self, x):
        return tf.nn.relu(x)

    def dropout(self, x, keep_probability, noise_shape=None):
        if noise_shape is not None:
            oup = tf.nn.dropout(x=x, keep_prob=keep_probability, noise_shape=noise_shape)
        else:
            oup = tf.nn.dropout(x=x, keep_prob=keep_probability)
        return oup

    def maxpool(self, input, conf={}):
        # 池化层
        oup, _ = max_pool_argmax(input, conf)
        self.stru.append('pool_%s'%conf['ksize'][1])
        return oup

    def CONV(self, input, conf, trainable = None):
        if trainable is None:
            trainable = True
        oup = conv2d(input, conf, trainable=trainable, core_init=self.conv_init.get(self.conv_i))
        self.conv_i= self.conv_i+1
        self.stru.append('conv1d')
        return oup

    def BN(self, input, conf, trainable=None):
        # batch Normaization,
        if trainable is None:
            trainable = True
        oup = BN(input, conf, para=self.bn_init.get(self.bn_i), trainable=trainable)
        self.bn_i = self.bn_i+1
        self.stru.append('BN')
        return oup

    def NN(self, input, conf, trainable=None):
        if trainable is None:
            trainable = True
        oup = NN(input, conf, trainable, core_init = self.nn_init.get(self.nn_i))
        self.nn_i = self.nn_i+1
        return oup


