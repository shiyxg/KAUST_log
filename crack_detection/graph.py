import tensorflow as tf
from unit.conv import *
from unit.deconv import *
from unit.pool import *
from unit.BN import *
import os

class FCN(object):
    def __init__(self, input_shape=[64, 64]):
        self.graph = tf.Graph() # graph
        self.feed = [] # feed para
        self.train = None # 训练参数
        self.INPUT_SHAPE = input_shape +[3] # 输入
        self.OUTPUT_SHAPE = input_shape +[1] # 输入
        self.para = []
        self.loss = None
        self.trainable = True # 参数是否可训练

        # 中间结果名字储存
        self.conv_result = []
        self.deconv_result = []

        # 网络参数名储存
        self.conv = []
        self.deconv = []
        self.bn = []
        self.pool = []
        self.nn = []

        # stored para space
        self.conv_init = {}
        self.deconv_init = {}
        self.nn_init={}
        self.bn_init = {}
        self.stru = []
        self.bn_i = 0
        self.conv_i = 0
        self.deconv_i = 0
        self.nn_i = 0

        # the class num of FCN's output
        self.output_chn = None
        self.conv_trainable = True
        self.input_chn = 3

        self.vgg16 = np.load('./DATA/vgg16_weights.npz')

    def build_train_graph(self, NUM=100):

        return self.build_graph(NUM=NUM)

    def fcn_network(self, NUM):

        [images, labels_r, learning_rate, is_training, keep] = self.feed
        conf1 = {'is_training': is_training}
        conf2 = {'filterSize': None, 'outputChn': None, 'strideSize': [1]}

        conv_trainable = self.conv_trainable
        # with tf.name_scope('BN'):
        #     images = self.BN(images, conf1, trainable=tf.constant(True))
        # 第一层卷积
        with tf.name_scope('conv1'):
            conf2['filterSize'] = [3, 3]
            conf2['outputChn'] = 32
            x = self.CONV(images, conf2, trainable=conv_trainable)
            x = self.BN(x, conf1, trainable=conv_trainable)
            x = tf.nn.relu(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool1', 'ksize': [1,2,2,1]})

        # 第二层卷积
        with tf.name_scope('conv2'):
            conf2['filterSize'] = [3, 3]
            conf2['outputChn'] = 64
            x = self.CONV(x, conf2, trainable=conv_trainable)
            x = self.BN(x, conf1, trainable=conv_trainable)
            x = tf.nn.relu(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool2', 'ksize': [1, 2, 2, 1]})

        # 第三层卷积
        with tf.name_scope('conv3'):
            conf2['filterSize'] = [3, 3]
            conf2['outputChn'] = 128
            x = self.CONV(x, conf2, trainable=conv_trainable)
            x = self.BN(x, conf1, trainable=conv_trainable)
            x = tf.nn.relu(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool3', 'ksize': [1, 2, 2, 1]})

        # 第四次卷积
        with tf.name_scope('conv4'):
            conf2['filterSize'] = [3, 3]
            conf2['outputChn'] = 256
            x = self.CONV(x, conf2, trainable=conv_trainable)
            x = self.BN(x, conf1, trainable=conv_trainable)
            x = tf.nn.relu(x)
        self.conv_result.append(x)

        #x = self.maxpool(x, {'name': 'pool4', 'ksize': [1, 2, 2, 1]})
        # # 第五次卷积
        # with tf.name_scope('conv5'):
        #     conf2['filterSize'] = [3, 3]
        #     conf2['outputChn'] = 256
        #     x = self.CONV(x, conf2, trainable=conv_trainable)
        #     x = self.BN(x, conf1, trainable=conv_trainable)
        #     x = tf.nn.relu(x)
        # self.conv_result.append(x)

        conf3 = {
            'name': 'deconv',
            'scale': 1,
            'outputChn': self.output_chn,
            'outputSize': [self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]]
        }
        # 第一层反卷积
        with tf.name_scope('deconv1'):
            conf3['scale'] = 1
            x = self.DECONV(self.conv_result[0], conf3, trainable=True)
            x = self.BN(x, conf1)
            x = tf.nn.relu(x)
        self.deconv_result.append(x)

        # 第二层反卷积
        with tf.name_scope('deconv2'):
            conf3['scale'] = 2
            x = self.DECONV(self.conv_result[1], conf3, trainable=True)
            x = self.BN(x, conf1)
            x = tf.nn.relu(x)
        self.deconv_result.append(x)

        # 第三层反卷积
        with tf.name_scope('deconv3'):
            conf3['scale'] = 4
            x = self.DECONV(self.conv_result[2], conf3, trainable=True)
            x = self.BN(x, conf1)
            x = tf.nn.relu(x)
        self.deconv_result.append(x)

        # 第四层反卷积
        with tf.name_scope('deconv4'):
            conf3['scale'] = 8
            x = self.DECONV(self.conv_result[3], conf3, trainable=True)
            x = self.BN(x, conf1)
            x = tf.nn.relu(x)
        self.deconv_result.append(x)

        # # 第五层反卷积
        # with tf.name_scope('deconv5'):
        #     conf3['scale'] = 16
        #     x = self.DECONV(self.conv_result[4], conf3, trainable=True)
        #     x = self.BN(x, conf1)
        #     x = tf.nn.relu(x)
        # self.deconv_result.append(x)

        # 融合成1个四通道的输出
        with tf.name_scope('Fuse'):
            deconvFuse = tf.concat(self.deconv_result, 3, name='FuseTo1Imag')
            conf2['filterSize'] = [1]
            conf2['outputChn'] = 2
            x = self.CONV(deconvFuse, conf2, trainable=True)
            x = tf.nn.softmax(x, -1)
            x = tf.slice(x, [0,0,0,1],[-1]+self.OUTPUT_SHAPE)
        return x

    def build_graph(self, NUM=100):

        # setting the graph's output chn
        # when graph's output is 4, it could deal the question of distinguishing 3 tremors and 1 noise
        if self.output_chn is None:
            self.output_chn = 1
        with self.graph.as_default():
            # input
            INPUT_SHAPE = self.INPUT_SHAPE
            with tf.name_scope('ImagesLabels'):
                images = tf.placeholder(tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], self.input_chn])
                labels_r = tf.placeholder(tf.float32, shape=[None, INPUT_SHAPE[0], INPUT_SHAPE[1], self.output_chn])
                labels = labels_r
            learning_rate = tf.placeholder(tf.float32, name='LR')
            keep = tf.placeholder(tf.float32, name='keep')
            is_training = tf.placeholder(tf.bool, name='BN_controller')
            self.feed = [images, labels_r, learning_rate, is_training, keep]

            self.trainable = True
            x = self.fcn_network(NUM)

            # get collection of weight, BN, deconvweight et.al
            self.collection()

            loss = self.loss_define(x, labels)

            # 训练与其他处理
            with tf.name_scope('other'):
                train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
                result = x
                self.x = result

            summary = self.summary(loss, x)

            self.train = train
            self.para = [loss, result, summary]

    def loss_define(self, x, labels):
        with tf.name_scope('WightCrossEntro'):
            cost = tf.nn.weighted_cross_entropy_with_logits(labels, x, 10)
            loss = tf.reduce_mean(cost)
            self.loss = loss
        return loss

    def summary(self, loss, x):
        with tf.name_scope('summary'):
            tf.summary.scalar('loss', tf.reduce_mean(loss))
            model_result = tf.cast(self.feed[1], tf.float32)
            label_result = tf.round(x)

            tf.summary.image('input', self.feed[0])
            tf.summary.image('FCN_result', label_result)
            tf.summary.image('label', model_result)
            for i in range(len(self.deconv_result)):
                tf.summary.image('deconv%s' % i, tf.slice(self.deconv_result[i],
                                                          [0,0,0,0],[1,self.deconv_result[i].shape[1], self.deconv_result[i].shape[2], 1]))

            for i in range(len(self.conv)):
                tf.summary.histogram('conv%s' % i, self.conv[i])
            for i in range(len(self.deconv)):
                tf.summary.histogram('deconv%s' % i, self.deconv[i])

            tf.summary.histogram('deconv_last', self.deconv_result[-1])
            summary = tf.summary.merge_all()
        return summary

    def collection(self):
        self.conv = tf.get_collection('conv_core')
        self.deconv = tf.get_collection('deconv_cores')
        for i in range(len(tf.get_collection('shift'))):
            self.bn.append(
                [tf.get_collection('shift')[i], tf.get_collection('scale')[i], tf.get_collection('moving_mean')[i],
                 tf.get_collection('moving_var')[i]])
        for i in range(len(tf.get_collection('w'))):
            self.nn.append(
                [tf.get_collection('w')[i], tf.get_collection('b')[i]]
            )

    def save_para_fcn(self, sess, file_name, NUM):

        c = file_name.split('/')
        c.pop(-1)
        path = ''
        for i in c:
            path = path + '/' + i
        path = path[1:]
        if not os.path.exists(path):
            os.mkdir(path)

        feedData = {self.feed[0]: np.zeros([NUM, self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], self.input_chn]).astype('float32'),
                    self.feed[1]: np.zeros([NUM, self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], self.output_chn]).astype('float32'),
                    self.feed[2]: 1e-3, self.feed[3]: False, self.feed[4]: 1}
        fetchVariables = [self.conv, self.bn, self.deconv, self.nn]
        # 获取网络参数
        [cores, bn, decores, nn] = sess.run(fetches=fetchVariables, feed_dict=feedData)

        # 保存网络参数
        print('*******restoreing para*********')
        dic = {'conv_init': {}, 'bn_init': {}, 'deconv_init': {}, 'nn_init':{}}
        for i in range(len(cores)):
            dic['conv_init'][i] = np.array(cores[i])
        for i in range(len(bn)):
            dic['bn_init'][i] = np.array(bn[i])
        for i in range(len(decores)):
            dic['deconv_init'][i] = np.array(decores[i])
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
        self.deconv_init = para_dict['deconv_init']
        self.nn_init = para_dict['nn_init']

        # print(self.conv_init[2])
        # for i in range(3):
        #     self.conv_init[2][0,0,i,:] = self.conv_init[2][0,0,i,:] * 0
        # print(self.conv_init[2])
        # when store CONV para from CNN
        if FROMCNN:
            for i in self.bn_init.keys():
                if i not in self.conv_init.keys():
                    self.bn_init[i] = None
                    print('layer', i, 'BN_para is set to None')

    def maxpool(self, input, conf={}):
        # 池化层
        oup, _ = max_pool_argmax(input, conf)
        self.stru.append('pool_%s'%conf['ksize'][1])
        return oup

    def CONV(self, input, conf, trainable = None):
        # 一维卷积
        if trainable is None:
            trainable = self.trainable
        oup = conv2d(input, conf, trainable=trainable, core_init=self.conv_init.get(self.conv_i))
        self.conv_i= self.conv_i+1
        self.stru.append('conv1d')
        return oup

    def DECONV(self, input, conf, use_bilinear=False, trainable=None):
        # 反卷积，用于FCN的pixeltopixel输出
        if trainable is None:
            trainable = self.trainable
        oup = deconv2d(input, conf, use_bilinear=use_bilinear, trainable=trainable, core_init=self.deconv_init.get(self.deconv_i))
        self.deconv_i = self.deconv_i+1
        self.stru.append('deconv2d')
        return oup

    def BN(self, input, conf, trainable=None):
        # batch Normaization, 网络输出
        if trainable is None:
            trainable = self.trainable
        oup = BN(input, conf, para=self.bn_init.get(self.bn_i), trainable=trainable)
        self.bn_i = self.bn_i+1
        self.stru.append('BN')
        return oup

class FCN_VGG16(FCN):

    def fcn_network(self, NUM):
        [images, labels_r, learning_rate, is_training, keep] = self.feed
        conf1 = {'is_training': is_training}
        conf2 = {'filterSize': None, 'outputChn': None, 'strideSize': [1]}

        conv_trainable = self.conv_trainable

        # conv 1-VGG
        with tf.name_scope('conv1-1'):
            w  = tf.constant(self.vgg16['conv1_1_W'], name='VGG16W')
            b  = tf.constant(self.vgg16['conv1_1_b'], name='VGG16B')
            x = tf.nn.conv2d(images, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        with tf.name_scope('conv1-2'):
            w  = tf.constant(self.vgg16['conv1_2_W'], name='VGG16W')
            b  = tf.constant(self.vgg16['conv1_2_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool1', 'ksize': [1,2,2,1]})

        # conv 2-VGG
        with tf.name_scope('conv2-1'):
            w  = tf.constant(self.vgg16['conv2_1_W'], name='VGG16W')
            b  = tf.constant(self.vgg16['conv2_1_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        with tf.name_scope('conv2-2'):
            w  = tf.constant(self.vgg16['conv2_2_W'], name='VGG16W')
            b  = tf.constant(self.vgg16['conv2_2_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool2', 'ksize': [1,2,2,1]})

        # conv 3-VGG
        with tf.name_scope('conv3-1'):
            w  = tf.constant(self.vgg16['conv3_1_W'], name='VGG16W')
            b  = tf.constant(self.vgg16['conv3_1_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        with tf.name_scope('conv3-2'):
            w  = tf.constant(self.vgg16['conv3_2_W'], name='VGG16W')
            b  = tf.constant(self.vgg16['conv3_2_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        with tf.name_scope('conv3-3'):
            w  = tf.constant(self.vgg16['conv3_3_W'], name='VGG16W')
            b  = tf.constant(self.vgg16['conv3_3_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool3', 'ksize': [1,2,2,1]})

        # by my self
        with tf.name_scope('conv4-1-me'):
            conf2['filterSize'] = [5, 5]
            conf2['outputChn'] = 256
            x = self.CONV(x, conf2, trainable=conv_trainable)
            x = self.BN(x, conf1, trainable=conv_trainable)
            x = tf.nn.relu(x)
        with tf.name_scope('conv4-2-me'):
            conf2['filterSize'] = [5, 5]
            conf2['outputChn'] = 256
            x = self.CONV(x, conf2, trainable=conv_trainable)
            x = self.BN(x, conf1, trainable=conv_trainable)
            x = tf.nn.relu(x)
        self.conv_result.append(x)

        # 反卷积部分,这是简单的FCN
        conf3 = {
            'name': 'deconv',
            'scale': 1,
            'outputChn': self.output_chn,
            'outputSize': [self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]]
        }
        scale = {1:1, 2:2, 3:4, 4:8}
        # 反卷积层循环
        for i in range(len(self.conv_result)):
            with tf.name_scope('deconv-me%s'%(i+1)):
                conf3['scale'] = scale[i+1]
                x = self.DECONV(self.conv_result[i], conf3, trainable=True)
                x = self.BN(x, conf1)
                x = tf.nn.relu(x)
            self.deconv_result.append(x)
        # 融合成1个四通道的输出
        with tf.name_scope('Fuse'):
            deconvFuse = tf.concat(self.deconv_result, 3, name='FuseTo1Imag')
            conf2['filterSize'] = [1]
            conf2['outputChn'] = 2
            x = self.CONV(deconvFuse, conf2, trainable=True)
            x = tf.nn.softmax(x, -1)
            x = tf.slice(x, [0,0,0,1],[-1]+self.OUTPUT_SHAPE)
        return x


class FCN_UNET(FCN):
    def fcn_network(self, NUM):
        [images, labels_r, learning_rate, is_training, keep] = self.feed
        conf1 = {'is_training': is_training}
        conf2 = {'filterSize': None, 'outputChn': None, 'strideSize': [1]}

        conv_trainable = self.conv_trainable

        # conv 1-VGG
        with tf.name_scope('conv1-1'):
            w = tf.constant(self.vgg16['conv1_1_W'], name='VGG16W')
            b = tf.constant(self.vgg16['conv1_1_b'], name='VGG16B')
            x = tf.nn.conv2d(images, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        with tf.name_scope('conv1-2'):
            w = tf.constant(self.vgg16['conv1_2_W'], name='VGG16W')
            b = tf.constant(self.vgg16['conv1_2_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool1', 'ksize': [1, 2, 2, 1]})

        # conv 2-VGG
        with tf.name_scope('conv2-1'):
            w = tf.constant(self.vgg16['conv2_1_W'], name='VGG16W')
            b = tf.constant(self.vgg16['conv2_1_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        with tf.name_scope('conv2-2'):
            w = tf.constant(self.vgg16['conv2_2_W'], name='VGG16W')
            b = tf.constant(self.vgg16['conv2_2_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool2', 'ksize': [1, 2, 2, 1]})

        # conv 3-VGG
        with tf.name_scope('conv3-1'):
            w = tf.constant(self.vgg16['conv3_1_W'], name='VGG16W')
            b = tf.constant(self.vgg16['conv3_1_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        with tf.name_scope('conv3-2'):
            w = tf.constant(self.vgg16['conv3_2_W'], name='VGG16W')
            b = tf.constant(self.vgg16['conv3_2_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        with tf.name_scope('conv3-3'):
            w = tf.constant(self.vgg16['conv3_3_W'], name='VGG16W')
            b = tf.constant(self.vgg16['conv3_3_b'], name='VGG16B')
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
            x = tf.add(x, b)
            x = tf.nn.relu(x)
        self.conv_result.append(x)
        x = self.maxpool(x, {'name': 'pool3', 'ksize': [1, 2, 2, 1]})

        # by my self
        with tf.name_scope('conv4-1-me'):
            conf2['filterSize'] = [5, 5]
            conf2['outputChn'] = 256
            x = self.CONV(x, conf2, trainable=conv_trainable)
            x = self.BN(x, conf1, trainable=conv_trainable)
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, keep_prob=keep, noise_shape=[1, 1, 1, 256])
        with tf.name_scope('conv4-2-me'):
            conf2['filterSize'] = [5, 5]
            conf2['outputChn'] = 256
            x = self.CONV(x, conf2, trainable=conv_trainable)
            x = self.BN(x, conf1, trainable=conv_trainable)
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, keep_prob=keep, noise_shape=[1, 1, 1, 256])

        self.conv_result.append(x)

        conf3 = {
            'name': 'deconv',
            'scale': 1,
            'outputChn': self.output_chn,
            'outputSize': [self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]]
        }

        with tf.name_scope('deconv-me2'):
            conf3['scale'] = 2
            conf3['outputChn'] = 64
            conf3['outputSize'] = [self.INPUT_SHAPE[0]//4, self.INPUT_SHAPE[1]//4]
            x = self.DECONV(x, conf3, trainable=True)
            x = self.BN(x, conf1)
            x = tf.nn.relu(x)
            self.deconv_result.append(x)
        with tf.name_scope('deconv-me4'):
            conf3['scale'] = 2
            conf3['outputChn'] = 32
            conf3['outputSize'] = [self.INPUT_SHAPE[0]//2, self.INPUT_SHAPE[1]//2]
            x = self.DECONV(x, conf3, trainable=True)
            x = self.BN(x, conf1)
            x = tf.nn.relu(x)
            self.deconv_result.append(x)
        with tf.name_scope('deconv-me8'):
            conf3['scale'] = 2
            conf3['outputChn'] = 32
            conf3['outputSize'] = [self.INPUT_SHAPE[0], self.INPUT_SHAPE[1]]
            # combine convresult
            x = tf.concat([self.conv_result[1], x], 3, name='combineCONV')
            x = self.DECONV(x, conf3, trainable=True)
            x = self.BN(x, conf1)
            x = tf.nn.relu(x)
            self.deconv_result.append(x)

        with tf.name_scope('conv-last_2'):
            conf2['filterSize'] = [3, 3]
            conf2['outputChn'] = 32
            x = tf.concat([self.conv_result[0], x], 3, name='combineCONV')
            x = self.CONV(x, conf2, trainable=conv_trainable)
            x = self.BN(x, conf1)
            x = tf.nn.relu(x)

        with tf.name_scope('conv-last'):
            conf2['filterSize'] = [5, 5]
            conf2['outputChn'] = 2
            # combine convresult
            x = self.CONV(x, conf2, trainable=conv_trainable)
            x = tf.nn.softmax(x, -1)
            x = tf.slice(x, [0,0,0,1],[-1]+self.OUTPUT_SHAPE)
            self.deconv_result.append(x)
        return x
