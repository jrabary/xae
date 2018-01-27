import tensorflow as tf
import numpy as np
import PIL.Image as Image
from xae.data import mnist_dataset

tfgan = tf.contrib.gan
ds = tf.contrib.distributions

default_params = tf.contrib.training.HParams(
    latent_space_dim=2,
    observable_space_dims=[28, 28, 1],
    learning_rate=1e-4
)


def decode(z, observable_space_dims):
    with tf.variable_scope('Decoder', [z]):
        logits = tf.layers.dense(z, 200, activation=tf.nn.tanh)
        logits = tf.layers.dense(logits, np.prod(observable_space_dims))
        return logits


def encoder(x, observable_space_dim, latent_dim):

    with tf.variable_scope('Encoder', [x]):
        x = tf.reshape(x, [-1, np.prod(observable_space_dim)])
        h = tf.layers.dense(x, 10, activation=tf.nn.tanh)
        mu = tf.layers.dense(h, latent_dim)
        sigma_sq = tf.layers.dense(h, latent_dim)

        return mu, sigma_sq


def input_fn():
    dataset = (mnist_dataset.train('data/mnist')
               .repeat()
               .cache()
               .shuffle(buffer_size=50000)
               .batch(128)
               )
    (images, _) = dataset.make_one_shot_iterator().get_next()

    images = tf.reshape(images, [128, 28, 28, 1])

    return images, images


def model_fn(features, labels, mode, params):
    x = features

    q_mu, q_cov = encoder(x, params.observable_space_dims, params.latent_space_dim)
    q_z_given_x = ds.MultivariateNormalDiag(q_mu, q_cov)

    z_samples = q_z_given_x.sample()

    p_x_given_z_logits = decode(z_samples, params.observable_space_dims)
    p_x_given_z = ds.Bernoulli(logits=p_x_given_z_logits)
    x_mean = p_x_given_z.mean()
    reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
    tf.summary.image('reconstruction/x_mean', reconstruction)


    # compute loss
    # prior := p_z
    prior = ds.MultivariateNormalDiag(loc=tf.zeros([1, params.latent_space_dim], dtype=tf.float32),
                                      scale_diag=tf.ones([1, params.latent_space_dim], dtype=tf.float32))

    # KL can be seen as regularization term!
    KL = ds.kl_divergence(q_z_given_x, prior)

    # The ELBO = reconstruction term + regularization term
    elbo = tf.reduce_sum(tf.reduce_sum(p_x_given_z.log_prob(labels), axis=[1, 2, 3]) - KL)
    loss = - elbo

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='training', params=default_params)
    estimator.train(input_fn=input_fn)
