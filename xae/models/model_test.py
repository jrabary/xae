import tensorflow as tf
from math import log

from xae.models.celebs import encode

num_layers = 4
num_units = 1024

for i in range(num_layers):
    scale = 2**(num_layers - i - 1)
    print(scale, num_units//scale)


depth = 128
inp_shape = 64
for i in range(int(log(inp_shape, 2))):
    scope = 'conv%i' % (i + 1)
    current_depth = depth * 2**i
    print('current depth', current_depth)


x = tf.random_uniform([10, 64, 64, 3], dtype=tf.float32)

z = encode(x, 64)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    z_val = sess.run(z)

    print(z_val[0].shape)