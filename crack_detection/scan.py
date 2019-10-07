# -*- coding: utf-8 -*-

from __future__ import print_function

# Graph is built in Graph.py
import tensorflow as tf
import sys
import numpy as np
import sys
import scipy.io as io
import os
import matplotlib as mpl
from pylab import *
from time import time
from graph import *
from get_samples import DATA
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

kk=0
SCAN=False
NUM=1
image = Image.open('samples_big/%02d.jpg'%kk)
# image = Image.open('DATA/vug.jpg')
# image = image.rotate(90, expand=True)
# image = Image.open('drones/01.jpg')
image = image.resize([int(image.size[0]/2), int(image.size[1]/2)])

data = np.array(image).astype('float32')/128-1
shape = data.shape
h, w, c = shape

# g = FCN([256, 256])
if not SCAN:
    g = FCN_UNET([h,w])
else:
    g = FCN_UNET([256,256])
g.output_chn = 1
g.input_chn = 3
g.restore_para('C:/Users/shiyx/Documents/ML_log/crack_detection/VGG/test_UNET06/para_npy/9501.npy')
g.build_graph(NUM)
threshold = 0.1

# image = np.load('samples_big/%02d_gee.npy'%kk)
# image = np.transpose(image.reshape([image.shape[0], 3, image.shape[1]//3]), [0,2,1])
# image[:,:,1] = image[:,:,1]*128+128
# image[:, :, 2] = image[:, :, 2] * 128+128
# a = image[:,:,0]
# # a[where(image[:,:,1]==255)]=0
# # a[where(image[:,:,2]==255)]=0
# image[:,:,0] = a
# data = image
# data = data/128-1
# imshow(data)



# shape = [256,256,3]
# h, w, c = shape
# data = np.array(image).astype('float32')[3129:3129+h, 1322+256:1322+256+w, :]
def FCN_scan(data, g):
    if not SCAN:
        with tf.Session(graph=g.graph) as sess:
            ops = tf.get_collection('ops')
            sess.run(tf.global_variables_initializer())

            LR = 1e-3
            for i in range(1):
                print(data.shape)
                print([h,w])
                trainBatch = [data.reshape([1, h, w, c]), np.ones([1, h, w, 1]).astype('float32')]
                feedData = {g.feed[0]: trainBatch[0], g.feed[1]: trainBatch[1],
                            g.feed[2]: LR, g.feed[3]: True , g.feed[4]: 1}
                fetchVariables = g.para[1]
                if np.isnan(trainBatch[0].max()):
                    raise ValueError('1')
                result = sess.run(fetches=fetchVariables, feed_dict=feedData)
                result = result[0,:,:,0]
                r = np.zeros_like(result)
                r[:,:] = result
                r[np.where(r<threshold)]=None
                r[0,0] = 0
                figure()
                imshow((data+1)*128/255)
                imshow(r, cmap='bwr')
                # subplot(122)
                # imshow(data / 255)
                # np.save('test02_result.npy', np.array(result))

    else:
        result = np.zeros([h,w])
        with tf.Session(graph=g.graph) as sess:

            ops = tf.get_collection('ops')
            sess.run(tf.global_variables_initializer())

            LR = 1e-3
            for i in range(1):
                j,k=[0, 0]
                scale = data.sum(axis=2).max()
                print(scale)
                while 1:
                    inp = data[j:j+256, k:k+256,:]
                    # inp = inp.sum(axis=2)/scale
                    inp = np.reshape(inp, [1,256,256,g.input_chn]).astype('float32')

                    oup = np.zeros([1,256,256,1]).astype('float32')
                    trainBatch = [inp, oup]
                    feedData = {g.feed[0]: trainBatch[0], g.feed[1]: trainBatch[1],
                                g.feed[2]: LR, g.feed[3]: False, g.feed[4]: 1}
                    fetchVariables = g.para[1]
                    result_jk = sess.run(fetches=fetchVariables, feed_dict=feedData)
                    # result_jk[np.where(result_jk < 0.94)] = None
                    result[j:j+256, k:k+256] = result_jk[0, :, :, 0]

                    j = j+256
                    if j>=h:
                        j = 0
                        k = k+256
                    elif h-256<j<h:
                        j = h-256

                    if k>=w:
                        break
                    elif w-256<=k<w:
                        k=w-256
                    print('\r %04d/%d'%(k, w), end='')

                # io.savemat('GEE-epoch1-32',  {'result':result.astype('float32')})
                r = np.zeros_like(result)
                r[:, :] = result
                r[where(result<threshold)]=None
                r[0,0]=0

                # imshow(data[:,:,0])
                figure()
                imshow(data[:,:,0], cmap='gray')
                imshow(r, cmap='bwr')
                colorbar()
    save('temp.npy', result.astype('float32'))
    show()
    plt.close('all')
FCN_scan(data, g)



