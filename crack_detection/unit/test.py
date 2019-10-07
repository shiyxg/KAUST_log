import tensorflow as tf
import numpy as np
from unit.pool import *
a = tf.Variable(np.random.random([1, 3, 3, 1]))
b, argmax = max_pool_argmax(a, conf={})
c = max_unpool_2d(b, argmax, conf={'output_shape':[3,3,1]})

o = tf.ones_like(argmax)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    inp, out, arg, ones, re = sess.run([a, b, argmax, o, c])

print(inp[0,:,:,0])
print('*************************')
print(out[0,:,:,0])
print('*************************')
print(arg[0,:,:,0])
print('*************************')
print(ones[0,:,:,0])
print('*************************')
print(re[0,:,:,0])