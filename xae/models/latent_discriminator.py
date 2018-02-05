"""Discriminator on the latent space"""

import tensorflow as tf


def discriminator_fn(inputs):
    """

    Args:
        inputs: Real or generated inputs

    Returns:
        logits for the probability that the inputs are reals.
    """
    num_units = 1024
    num_layers = 4
    nowozin_trick = False # True

    # No convolutions as GAN happens in the latent space
    with tf.variable_scope('z_adversary', values=[inputs]):
        hi = inputs
        for i in range(num_layers):
            hi = tf.layers.dense(hi, num_units, activation=tf.nn.relu6, name='h{}_lin'.format(i + 1))
        logits = tf.layers.dense(hi, 1, name='hfinal_lin', activation=None)
        # if nowozin_trick:
        #     # We are doing GAN between our model Qz and the true Pz.
        #     # Imagine we know analytical form of the true Pz.
        #     # The optimal discriminator for D_JS(Pz, Qz) is given by:
        #     # Dopt(x) = log dPz(x) - log dQz(x)
        #     # And we know exactly dPz(x). So add log dPz(x) explicitly
        #     # to the discriminator and let it learn only the remaining
        #     # dQz(x) term. This appeared in the AVB paper.
        #     # assert opts['pz'] == 'normal', \
        #     #     'The GAN Pz trick is currently available only for Gaussian Pz'
        #     sigma2_p = POT_PZ_SCALE ** 2
        #     normsq = tf.reduce_sum(tf.square(inputs), 1)
        #     hi = hi - normsq / 2. / sigma2_p \
        #          - 0.5 * tf.log(2. * np.pi) \
        #          - 0.5 * ZDIM * np.log(sigma2_p)
    return logits
