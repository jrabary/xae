import functools
import collections
import tensorflow as tf
import numpy as np
import PIL.Image as Image
from xae.data import celeba_dataset
from xae.models import celebs
from xae.models.dcgan_generator import DCGANGenerator
from xae.models.latent_discriminator import discriminator_fn as z_discriminator_fn

tfgan = tf.contrib.gan
ds = tf.contrib.distributions
slim = tf.contrib.slim
training = tf.contrib.training

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
    gamma=10.,  # regularization hyper parameters
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

class WAEModel(collections.namedtuple('WAEModel',
                                      ('encoder', 'decoder', 'latent_discriminator'))):
    """ WAE Model """


class WAEGanTrainOps(collections.namedtuple('WAEGANTrainOps',
                                            ('ae_train_op', 'latent_gan_train_op', 'global_step_inc'))):
    """WAE-GAN Train operators"""


def wae_gan_train_ops(
        model,
        ae_loss,
        latent_gan_loss,
        ae_optimizer,
        latent_gan_optimizer,
        check_for_unused_update_ops=True,
        # Optional args to pass directly to the `create_train_op`.
        **kwargs):
    """Returns GAN train ops.

    The highest-level call in TFGAN. It is composed of functions that can also
    be called, should a user require more control over some part of the GAN
    training process.

    Args:
      model: A WAEModel.
      loss: A GANLoss.
      generator_optimizer: The optimizer for generator updates.
      discriminator_optimizer: The optimizer for the discriminator updates.
      check_for_unused_update_ops: If `True`, throws an exception if there are
        update ops outside of the generator or discriminator scopes.
      **kwargs: Keyword args to pass directly to
        `training.create_train_op` for both the generator and
        discriminator train op.

    Returns:
      A GANTrainOps tuple of (generator_train_op, discriminator_train_op) that can
      be used to train a generator/discriminator pair.
    """
    # Create global step increment op.
    global_step = tf.train.get_or_create_global_step()
    global_step_inc = global_step.assign_add(1)

    ae_update_ops = []
    latent_dis_update_ops = []

    ae_global_step = None
    if isinstance(generator_optimizer,
                  sync_replicas_optimizer.SyncReplicasOptimizer):
        # TODO(joelshor): Figure out a way to get this work without including the
        # dummy global step in the checkpoint.
        # WARNING: Making this variable a local variable causes sync replicas to
        # hang forever.
        generator_global_step = variable_scope.get_variable(
            'dummy_global_step_generator',
            shape=[],
            dtype=global_step.dtype.base_dtype,
            initializer=init_ops.zeros_initializer(),
            trainable=False,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES])
        ae_update_ops += [ae_global_step.assign(global_step)]
    with tf.name_scope('ae_train'):
        ae_train_op = training.create_train_op(
            total_loss=ae_loss,
            optimizer=ae_optimizer,
            variables_to_train=model.encoder.get_variables() + model.decoder.get_variables(),
            global_step=ae_global_step,
            update_ops=ae_update_ops,
            **kwargs)

    latent_gan_global_step = None
    if isinstance(discriminator_optimizer,
                  sync_replicas_optimizer.SyncReplicasOptimizer):
        # See comment above `generator_global_step`.
        discriminator_global_step = variable_scope.get_variable(
            'dummy_global_step_discriminator',
            shape=[],
            dtype=global_step.dtype.base_dtype,
            initializer=init_ops.zeros_initializer(),
            trainable=False,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES])
        latent_dis_update_ops += [latent_gan_global_step.assign(global_step)]
    with tf.name_scope('latent_gran_train'):
        latent_gan_train_op = training.create_train_op(
            total_loss=latent_gan_loss,
            optimizer=latent_gan_optimizer,
            variables_to_train=model.latent_discriminator.get_variables(),
            global_step=latent_gan_global_step,
            update_ops=latent_dis_update_ops,
            **kwargs)

    return WAEGanTrainOps(ae_train_op, latent_gan_train_op, global_step_inc)


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    x = features

    q_z_given_x = celebs.encode(x, params.latent_space_dim)

    z_samples = q_z_given_x.sample()

    generator = DCGANGenerator(z_samples, params.generator, is_training)

    x_mean = tf.reshape(generator.mean, [params.batch_size] + params.observable_space_dims)
    x_mean.set_shape([params.batch_size] + params.observable_space_dims)

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


def wae_model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    x = features

    q_z_given_x = celebs.encode(x, params.latent_space_dim)

    z_from_q_z = q_z_given_x.sample()

    generator = DCGANGenerator(z_from_q_z, params.generator, is_training)

    x_mean = tf.reshape(generator.mean, [params.batch_size] + params.observable_space_dims)
    x_mean.set_shape([params.batch_size] + params.observable_space_dims)

    reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
    tf.summary.image('reconstruction/x_mean', reconstruction)

    # compute losses
    # JS(q_z, p_z) using GAN
    p_z = ds.MultivariateNormalDiag(loc=tf.zeros([1, params.latent_space_dim], dtype=tf.float32),
                                    scale_diag=tf.ones([1, params.latent_space_dim], dtype=tf.float32))
    z_from_p_z = p_z.sample()

    gan_model = tfgan.gan_model(
        generator_fn=tf.identity,
        discriminator_fn=lambda z, _: z_discriminator_fn(z),
        real_data=z_from_p_z,
        discriminator_scope='',
        generator_inputs=z_from_q_z)

    # Build the GAN loss.
    generator_loss = functools.partial(tfgan.losses.modified_generator_loss, label_smoothing=0.)
    discriminator_loss = functools.partial(tfgan.losses.modified_discriminator_loss, label_smoothing=0.)
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=generator_loss,
        discriminator_loss_fn=discriminator_loss)

    # The reconstruction loss
    reconstruction_loss = generator.reconstruction_loss(labels)

    # WAE objective = reconstruction_loss + lambda*D(q_z, p_z)
    wae_loss = reconstruction_loss + params.gamma * gan_loss.generator_loss

    # tf.summary.scalar('reconstruction/loss', reconstruction_loss)

    wae_optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    wae_train_op = wae_optimizer.minimize(wae_loss, tf.train.get_or_create_global_step())

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
                                                                      patch_size=celebs_params.observable_space_dims[
                                                                          0]))
