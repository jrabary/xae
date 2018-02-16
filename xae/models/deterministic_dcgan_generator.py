from abc import ABCMeta
import tensorflow as tf
from math import log

slim = tf.contrib.slim
ds = tf.contrib.distributions


class DeterministicDCGANGenerator(object):
    """
    Deterministic DCGAN Like generator.
    """
    __metaclass__ = ABCMeta

    def __init__(self, latent_tensors, params, is_training):
        self._scope = 'DCGANGenerator'
        self._is_training = is_training
        self._normalizer_fn = slim.batch_norm
        self._normalizer_fn_args = {
            'is_training': is_training,
            'zero_debias_moving_mean': True,
            'fused': True,
        }
        self._outputs = self._forward(latent_tensors, params)

    @property
    def mean(self):
        return self._outputs

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self._scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)

    def _forward(self, latent_tensors, params):

        final_size = params['final_size']
        depth = params['depth']
        num_outputs = params['num_outputs']

        latent_tensors.get_shape().assert_has_rank(2)
        if log(final_size, 2) != int(log(final_size, 2)):
            raise ValueError('`final_size` (%i) must be a power of 2.' % final_size)
        if final_size < 8:
            raise ValueError('`final_size` (%i) must be greater than 8.' % final_size)

        end_points = {}
        num_layers = int(log(final_size, 2)) - 1
        with tf.variable_scope(self._scope, values=[latent_tensors]):
            with slim.arg_scope([self._normalizer_fn], **self._normalizer_fn_args):
                with slim.arg_scope([slim.conv2d_transpose],
                                    normalizer_fn=self._normalizer_fn,
                                    stride=2,
                                    kernel_size=4):
                    net = tf.expand_dims(tf.expand_dims(latent_tensors, 1), 1)

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
                    scope = 'output'
                    net = slim.conv2d(
                        net,
                        num_outputs,
                        normalizer_fn=None,
                        activation_fn=tf.nn.tanh,
                        kernel_size=1,
                        stride=1,
                        padding='VALID',
                        scope=scope)
                    end_points[scope] = net

        return net

    def reconstruction_loss(self, targets):
        """
        Compute reconstruction loss.
        :param targets:
        :return:
        """
        with tf.name_scope('DCGANGeneratorReconstruction'):
            diff = tf.squared_difference(targets, self._outputs)

            loss = tf.reduce_sum(diff, axis=[1, 2, 3])
            loss = 0.05 * tf.reduce_mean(loss)
            tf.losses.add_loss(loss)

        return loss

