import tensorflow as tf

from xae.models.encoder import Encoder

slim = tf.contrib.slim
ds = tf.contrib.distributions


class DCGANEncoder(Encoder):
    """
    DCGAN Like generator with Multivariate Gaussian
    """

    def __init__(self, obs_tensors, params, is_training):
        super(DCGANEncoder, self).__init__(obs_tensors, params, is_training)

    def _compute_prob_z_given_x(self, obs_tensors, params):

        x = obs_tensors

        depth = params['depth']
        latent_space_dim = params['latent_space_dim']

        inp_shape = x.get_shape().as_list()[1]

        with tf.variable_scope(self._scope, values=[x]):
            with slim.arg_scope([self._normalizer_fn], **self._normalizer_fn_args):
                with slim.arg_scope([slim.conv2d],
                                    stride=2,
                                    kernel_size=4,
                                    activation_fn=tf.nn.leaky_relu):
                    net = x
                    # for i in range(int(log(inp_shape, 2))):
                    for i in range(4):
                        scope = 'conv%i' % (i + 1)
                        current_depth = depth * 2 ** i
                        normalizer_fn_ = None if i == 0 else self._normalizer_fn
                        net = slim.conv2d(
                            net, current_depth, normalizer_fn=normalizer_fn_, scope=scope)

                    net = slim.flatten(net)

                    means = slim.fully_connected(net,
                                                 latent_space_dim,
                                                 normalizer_fn=None,
                                                 activation_fn=None)

                    log_sigmas = slim.fully_connected(net,
                                                      latent_space_dim,
                                                      normalizer_fn=None,
                                                      activation_fn=tf.nn.softplus)

        q_z_given_x = ds.MultivariateNormalDiag(means, log_sigmas)

        return q_z_given_x
