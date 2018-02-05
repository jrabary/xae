import functools
import tensorflow as tf
import numpy as np

tfgan = tf.contrib.gan

LAMBDA = 1.
ZDIM = 64
POT_PZ_SCALE = 1.


def z_adversary(inputs, reuse=False):
    num_units = 1024
    num_layers = 4
    nowozin_trick = False # True
    # No convolutions as GAN happens in the latent space
    with tf.variable_scope('z_adversary', reuse=reuse):
        hi = inputs
        for i in range(num_layers):
            hi = tf.layers.dense(hi, num_units, activation=tf.nn.relu6, name='h{}_lin'.format(i + 1))
        hi = tf.layers.dense(hi, 1, name='hfinal_lin', activation=None)
        if nowozin_trick:
            # We are doing GAN between our model Qz and the true Pz.
            # Imagine we know analytical form of the true Pz.
            # The optimal discriminator for D_JS(Pz, Qz) is given by:
            # Dopt(x) = log dPz(x) - log dQz(x)
            # And we know exactly dPz(x). So add log dPz(x) explicitly
            # to the discriminator and let it learn only the remaining
            # dQz(x) term. This appeared in the AVB paper.
            # assert opts['pz'] == 'normal', \
            #     'The GAN Pz trick is currently available only for Gaussian Pz'
            sigma2_p = POT_PZ_SCALE ** 2
            normsq = tf.reduce_sum(tf.square(inputs), 1)
            hi = hi - normsq / 2. / sigma2_p \
                 - 0.5 * tf.log(2. * np.pi) \
                 - 0.5 * ZDIM * np.log(sigma2_p)
    return hi


def gan_penalty(sample_qz, sample_pz):
    # Pz = Qz test based on GAN in the Z space
    logits_Pz = z_adversary(sample_pz)
    logits_Qz = z_adversary(sample_qz, reuse=True)
    loss_Pz = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_Pz, labels=tf.ones_like(logits_Pz)))
    loss_Qz = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_Qz, labels=tf.zeros_like(logits_Qz)))
    loss_Qz_trick = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_Qz, labels=tf.ones_like(logits_Qz)))

    loss_adversary = LAMBDA * (loss_Pz + loss_Qz)
    # Non-saturating loss trick
    loss_match = loss_Qz_trick
    return (loss_adversary, logits_Pz, logits_Qz), loss_match


def main():
    batch_size = 64
    z_from_q = tf.random_uniform((batch_size, ZDIM), dtype=tf.float32)
    z_from_p = tf.random_uniform((batch_size, ZDIM))

    d_loss, g_loss = gan_penalty(z_from_q, z_from_p)

    # Build the generator and discriminator.
    gan_model = tfgan.gan_model(
        generator_fn=tf.identity,
        discriminator_fn=lambda x, unused: z_adversary(x, reuse=True),  # you define
        real_data=z_from_p,
        discriminator_scope='',
        generator_inputs=z_from_q)

    # Build the GAN loss.
    generator_loss = functools.partial(tfgan.losses.modified_generator_loss, label_smoothing=0.)
    discriminator_loss = functools.partial(tfgan.losses.modified_discriminator_loss, label_smoothing=0.)
    gan_loss = tfgan.gan_loss(
        gan_model,
        generator_loss_fn=generator_loss,
        discriminator_loss_fn=discriminator_loss)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(10):
            d_loss_val, g_loss_val, g_loss_val_2, d_loss_val_2 = sess.run([d_loss[0], g_loss, gan_loss.generator_loss, gan_loss.discriminator_loss])
            print('==============', i)
            print('wae discriminator loss', d_loss_val)
            print('tfgan discrimintor loss', d_loss_val_2)
            print('wea generator loss', g_loss_val)
            print('tfgan generator loss', g_loss_val_2)


if __name__ == '__main__':
    main()
