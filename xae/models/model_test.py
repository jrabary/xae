import tensorflow as tf
from math import log

from xae.models.celebs import encode, decode

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
z = tf.random_uniform([10, 64], dtype=tf.float32)

q_z_given_x = encode(x, 64)

p_x_given_z = decode(z, 64)

z_samples = q_z_given_x.sample()

x_samples = p_x_given_z.sample()

log_prob = p_x_given_z.log_prob(x_samples)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    z_val = sess.run(z_samples)

    print('z samples', z_val.shape)

    x_val = sess.run(x_samples)
    print('x samples', x_val.shape)

    log_prob_val = sess.run(log_prob)
    print('log_prob', log_prob_val.shape)


