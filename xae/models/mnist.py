import tensorflow as tf
import numpy as np

ds = tf.contrib.distributions


def decode(z, observable_space_dims):
    with tf.variable_scope('Decoder', [z]):
        logits = tf.layers.dense(z, 200, activation=tf.nn.tanh)
        logits = tf.layers.dense(logits, np.prod(observable_space_dims))

    p_x_given_z = ds.Bernoulli(logits=logits)
    return p_x_given_z


def encoder(x, observable_space_dim, latent_dim):

    with tf.variable_scope('Encoder', [x]):
        x = tf.reshape(x, [-1, np.prod(observable_space_dim)])
        h = tf.layers.dense(x, 10, activation=tf.nn.tanh)
        mu = tf.layers.dense(h, latent_dim)
        sigma_sq = tf.layers.dense(h, latent_dim)

    q_z_given_x = ds.MultivariateNormalDiag(mu, sigma_sq)
    return q_z_given_x
