import tensorflow as tf
import numpy as np
import PIL.Image as Image
from xae.data import celeba_dataset
from xae.models import celebs
from xae.models.dcgan_generator import DCGANGenerator

tfgan = tf.contrib.gan
ds = tf.contrib.distributions
slim = tf.contrib.slim

default_params = tf.contrib.training.HParams(
    latent_space_dim=2,
    observable_space_dims=[28, 28, 1],
    learning_rate=1e-4,
)

celebs_params = tf.contrib.training.HParams(
    generator={
        'final_size': 32,
        'depth': 64,
        'num_outputs': 3
    },
    latent_space_dim=64,
    observable_space_dims=[32, 32, 3],
    learning_rate=1e-4,
    batch_size=64,
    train_data='/Users/jaonary/Data/celebA/img_align_celeba/*.jpg',
)


# def input_fn():
#     dataset = (mnist_dataset.train('data/mnist')
#                .repeat()
#                .cache()
#                .shuffle(buffer_size=50000)
#                .batch(128)
#                )
#     (images, _) = dataset.make_one_shot_iterator().get_next()
#
#     images = tf.reshape(images, [128, 28, 28, 1])
#
#     return images, images


def model_fn(features, labels, mode, params):

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    x = features

    q_z_given_x = celebs.encode(x, params.latent_space_dim)

    z_samples = q_z_given_x.sample()

    generator = DCGANGenerator(z_samples, params.generator, is_training)

    x_mean = tf.reshape(generator.mean, [params.batch_size] + params.observable_space_dims)
    x_mean.set_shape([params.batch_size]+params.observable_space_dims)

    reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
    tf.summary.image('reconstruction/x_mean', reconstruction)


    # compute loss
    # prior := p_z
    prior = ds.MultivariateNormalDiag(loc=tf.zeros([1, params.latent_space_dim], dtype=tf.float32),
                                      scale_diag=tf.ones([1, params.latent_space_dim], dtype=tf.float32))

    # KL can be seen as regularization term!
    KL = ds.kl_divergence(q_z_given_x, prior)

    # The ELBO = reconstruction term + regularization term
    reconstruction_loss = generator.reconstruction_loss(labels)
    # tf.summary.scalar('reconstruction/loss', reconstruction_loss)

    # elbo = tf.reduce_sum(tf.reduce_sum(log_prob,) - KL)
    elbo = tf.reduce_sum(reconstruction_loss - KL)
    loss = -elbo

    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.estimator.RunConfig(save_summary_steps=10)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir='celeba_training_2',
                                       params=celebs_params,
                                       config=config)
    estimator.train(input_fn=lambda: celeba_dataset.image_file_inputs(celebs_params.train_data,
                                                                      batch_size=celebs_params.batch_size,
                                                                      patch_size=celebs_params.observable_space_dims[0]))
