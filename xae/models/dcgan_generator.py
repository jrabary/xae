import tensorflow as tf
from math import log
from xae.models.generator import Generator

slim = tf.contrib.slim
ds = tf.contrib.distributions


class DCGANGenerator(Generator):
    """
    DCGAN Like generator with Multivariate Gaussian
    """

    def __init__(self, latent_tensors, params, is_training):
        super(DCGANGenerator, self).__init__(latent_tensors, params, is_training)

    def _compute_prob_x_given_z(self, latent_tensors, params):

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

        p_x_given_z = ds.MultivariateNormalDiag(tf.reshape(means, [-1, final_size * final_size * num_outputs]),
                                                tf.reshape(log_sigmas, [-1, final_size * final_size * num_outputs]))
        return p_x_given_z

    def reconstruction_loss(self, targets):
        """
        Compute the log probability of the targets.
        :param targets:
        :return:
        """
        with tf.name_scope('reconstruction_loss'):
            log_prob = self.p_x_given_z.log_prob(slim.flatten(targets))

        return log_prob

