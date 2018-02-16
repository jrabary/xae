import functools
import collections
import tensorflow as tf
from tensorflow.contrib.gan.python.train import RunTrainOpsHook
import numpy as np
import PIL.Image as Image
from xae.data import celeba_dataset

from xae.models import DCGANGenerator, DeterministicDCGANGenerator
from xae.models.dcgan_encoder import DCGANEncoder
from xae.models.latent_discriminator import discriminator_fn as z_discriminator_fn

tfgan = tf.contrib.gan
ds = tf.contrib.distributions
slim = tf.contrib.slim
training = tf.contrib.training
framework = tf.contrib.framework

default_params = tf.contrib.training.HParams(
    latent_space_dim=2,
    observable_space_dims=[28, 28, 1],
    learning_rate=1e-4,
)

celebs_params = tf.contrib.training.HParams(

    encoder={
        'depth': 128,
        'latent_space_dim': 64,
    },

    generator={
        'final_size': 64,
        'depth': 64,
        'num_outputs': 3
    },
    gamma=10.,  # regularization hyper parameters
    latent_space_dim=64,
    observable_space_dims=[64, 64, 3],
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
        encoder_variables,
        decoder_variables,
        latent_discriminator_variables,
        ae_loss,
        latent_gan_loss,
        ae_optimizer,
        latent_gan_optimizer,
        **kwargs):
    """Returns WAE-GAN train ops.

    """
    # Create global step increment op.
    global_step = tf.train.get_or_create_global_step()
    global_step_inc = global_step.assign_add(1)

    ae_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    latent_dis_update_ops = []

    ae_global_step = framework.local_variable(tf.zeros((), dtype=tf.int64), name='ae_global_step')
    ae_update_ops += [ae_global_step.assign(global_step)]
    with tf.name_scope('ae_train'):
        ae_train_op = training.create_train_op(
            total_loss=ae_loss,
            optimizer=ae_optimizer,
            variables_to_train=encoder_variables + decoder_variables,
            global_step=ae_global_step,
            update_ops=ae_update_ops,
            **kwargs)

    latent_gan_global_step = framework.local_variable(tf.zeros((), dtype=tf.int64), name='latent_gan_global_step')
    latent_dis_update_ops += [latent_gan_global_step.assign(global_step)]
    with tf.name_scope('latent_gran_train'):
        latent_gan_train_op = training.create_train_op(
            total_loss=latent_gan_loss,
            optimizer=latent_gan_optimizer,
            variables_to_train=latent_discriminator_variables,
            global_step=latent_gan_global_step,
            update_ops=latent_dis_update_ops,
            **kwargs)

    return WAEGanTrainOps(ae_train_op, latent_gan_train_op, global_step_inc)


def get_sequential_train_hooks(wae_train_steps=1, latent_gan_train_steps=1):
    """Returns a hooks function for sequential WAE-GAN training.
    Args:
      wae_train_steps:

      latent_gan_train_steps:
    Returns:
      A function that takes a WAE-GANTrainOps tuple and returns a list of hooks.
    """

    def get_hooks(train_ops):
        wae_hook = RunTrainOpsHook(train_ops.ae_train_op, wae_train_steps)
        latent_gan_hook = RunTrainOpsHook(train_ops.latent_gan_train_op, latent_gan_train_steps)
        return [wae_hook, latent_gan_hook]

    return get_hooks


def model_fn(features, labels, mode, params):
    """ VAE model function.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    x = features

    encoder = DCGANEncoder(x, params.encoder, is_training)

    # q_z_given_x = celebs.encode(x, params.latent_space_dim)

    z_samples = encoder.sample()

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
    KL = ds.kl_divergence(encoder.q_z_given_x, prior)

    # The ELBO = reconstruction term + regularization term
    reconstruction_loss = generator.reconstruction_loss(labels)
    # tf.summary.scalar('reconstruction/loss', reconstruction_loss)

    # elbo = tf.reduce_sum(tf.reduce_sum(log_prob,) - KL)
    elbo = tf.reduce_mean(reconstruction_loss - KL)
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
    # latent space
    encoder = DCGANEncoder(x, params.encoder, is_training)
    z_from_q_z = encoder.sample()

    generator = DeterministicDCGANGenerator(z_from_q_z, params.generator, is_training)

    x_mean = tf.reshape(generator.mean, [params.batch_size] + params.observable_space_dims)
    x_mean.set_shape([params.batch_size] + params.observable_space_dims)

    reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
    tf.summary.image('reconstruction/x_mean', reconstruction)

    # compute losses
    # JS(q_z, p_z) using GAN
    p_z = ds.MultivariateNormalDiag(loc=tf.zeros([params.latent_space_dim], dtype=tf.float32),
                                    scale_diag=tf.ones([params.latent_space_dim], dtype=tf.float32))
    z_from_p_z = p_z.sample(params.batch_size)

    # Build the GAN loss.
    gan_model = tfgan.gan_model(
        generator_fn=tf.identity,
        discriminator_fn=lambda z, _: z_discriminator_fn(z),
        real_data=z_from_p_z,
        discriminator_scope='LatentDiscriminator',
        generator_inputs=z_from_q_z)
    generator_loss = functools.partial(tfgan.losses.modified_generator_loss, label_smoothing=0.)
    discriminator_loss = functools.partial(tfgan.losses.modified_discriminator_loss, label_smoothing=0.)
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=generator_loss,
        discriminator_loss_fn=discriminator_loss)

    # The reconstruction loss
    reconstruction_loss = generator.reconstruction_loss(labels)

    # WAE objective = reconstruction_loss + gamma*D(q_z, p_z)
    wae_loss = reconstruction_loss + params.gamma * gan_loss.generator_loss
    tf.summary.scalar('losses/reconstruction', reconstruction_loss)
    tf.summary.scalar('losses/penalty', gan_loss.generator_loss)
    tf.summary.scalar('losses/wae', wae_loss)

    # tf.summary.scalar('reconstruction/loss', reconstruction_loss)

    wae_optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    latent_gan_optimizer = tf.train.AdadeltaOptimizer(learning_rate=params.learning_rate)

    latent_dis_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'LatentDiscriminator')
    train_ops = wae_gan_train_ops(encoder.get_trainable_variables(),
                                  generator.get_trainable_variables(),
                                  latent_dis_variables,
                                  wae_loss,
                                  gan_loss.discriminator_loss,
                                  wae_optimizer,
                                  latent_gan_optimizer
                                  )

    training_hook_fn = get_sequential_train_hooks()

    training_hooks = training_hook_fn(train_ops)

    # Define Run Hooks

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=wae_loss,
        train_op=train_ops.global_step_inc,
        training_hooks=training_hooks
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.estimator.RunConfig(save_summary_steps=10)

    estimator = tf.estimator.Estimator(model_fn=wae_model_fn,
                                       model_dir='training/celeba_wae_gan',
                                       params=celebs_params,
                                       config=config)
    estimator.train(input_fn=lambda: celeba_dataset.image_file_inputs(celebs_params.train_data,
                                                                      batch_size=celebs_params.batch_size,
                                                                      patch_size=celebs_params.observable_space_dims[
                                                                          0]), steps=10000)
