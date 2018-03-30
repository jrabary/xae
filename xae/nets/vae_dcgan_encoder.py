import tensorflow as tf
from math import log
from xae.nets.vae_base import GaussianEncoder

ds = tf.contrib.distributions
slim = tf.contrib.slim


class VAEDCGanEncoder(GaussianEncoder):
    """
    DC GAN like encoder.
    """

    def __init__(self, params):
        super(VAEDCGanEncoder, self).__init__(params)

    def forward(self, x, is_training):
        depth = self._params['depth']
        latent_space_dim = self._params['latent_space_dim']

        inp_shape = x.get_shape().as_list()[1]

        with tf.variable_scope(self._scope, values=[x]):
            with slim.arg_scope(self.arg_scope(is_training)):
                with slim.arg_scope([slim.conv2d],
                                    stride=2,
                                    kernel_size=4,
                                    activation_fn=tf.nn.leaky_relu):
                    net = x
                    for i in range(int(log(inp_shape, 2))):
                        scope = 'conv%i' % (i + 1)
                        current_depth = depth * 2 ** i
                        if i == 0:
                            net = slim.conv2d(
                                net, current_depth, normalizer_fn=None, scope=scope)
                        else:
                            net = slim.conv2d(
                                net, current_depth, scope=scope)

                    net = slim.flatten(net)

                    means = slim.fully_connected(net,
                                                 latent_space_dim,
                                                 normalizer_fn=None,
                                                 activation_fn=None)

                    log_sigmas = slim.fully_connected(net,
                                                      latent_space_dim,
                                                      normalizer_fn=None,
                                                      activation_fn=tf.nn.softplus)

                    return means, log_sigmas
