import tensorflow as tf

from math import log
from xae.nets.vae_base import GaussianDecoder

ds = tf.contrib.distributions
slim = tf.contrib.slim


class VAEDCGanDecoder(GaussianDecoder):

    def __init__(self, params):
        super(VAEDCGanDecoder, self).__init__(params)

    def _mean_and_std(self, z, is_training):

        final_size = self._params['output_size']
        depth = self._params['depth']
        num_outputs = self._params['output_channels']

        z.get_shape().assert_has_rank(2)

        if log(final_size, 2) != int(log(final_size, 2)):
            raise ValueError('`final_size` (%i) must be a power of 2.' % final_size)
        if final_size < 8:
            raise ValueError('`final_size` (%i) must be greater than 8.' % final_size)

        end_points = {}
        num_layers = int(log(final_size, 2)) - 1
        with tf.variable_scope(self._scope, values=[z]):
            with slim.arg_scope([slim.conv2d_transpose],
                                normalizer_fn=self._normalizer_fn,
                                stride=2,
                                kernel_size=4):
                net = tf.expand_dims(tf.expand_dims(z, 1), 1)

                # First upscaling is different because it takes the input vector.
                current_depth = depth * 2 ** (num_layers - 1)
                scope = 'deconv1'
                net = slim.conv2d_transpose(
                    net, current_depth, stride=1, padding='VALID', scope=scope)
                end_points[scope] = net

                for i in range(2, num_layers):
                    scope = 'deconv%i' % (i)
                    current_depth = depth * 2 ** (num_layers - i)
                    net = slim.conv2d_transpose(net, current_depth, scope=scope)
                    end_points[scope] = net

                # Last layer has different normalizer and activation.
                scope = 'deconv%i' % (num_layers)
                net = slim.conv2d_transpose(
                    net, depth, normalizer_fn=None, activation_fn=None, scope=scope)
                end_points[scope] = net

                # Convert to proper channels.
                scope = 'means'
                means = slim.conv2d(
                    net,
                    num_outputs,
                    normalizer_fn=None,
                    activation_fn=None,
                    kernel_size=1,
                    stride=1,
                    padding='VALID',
                    scope=scope)
                end_points[scope] = means

                means.get_shape().assert_has_rank(4)
                means.get_shape().assert_is_compatible_with(
                    [None, final_size, final_size, num_outputs])

                scope = 'log_sigmas'
                log_sigmas = slim.conv2d(
                    net,
                    num_outputs,
                    normalizer_fn=None,
                    activation_fn=tf.nn.softplus,
                    kernel_size=1,
                    stride=1,
                    padding='VALID',
                    scope=scope)
                end_points[scope] = log_sigmas

                log_sigmas.get_shape().assert_has_rank(4)
                log_sigmas.get_shape().assert_is_compatible_with(
                    [None, final_size, final_size, num_outputs])

        return (tf.reshape(means, [-1, final_size * final_size * num_outputs]),
                tf.reshape(log_sigmas, [-1, final_size * final_size * num_outputs]))
