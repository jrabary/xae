"""VAE base classes definitions. """

from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from xae.nets.base_network import BaseNetwork

slim = tf.contrib.slim
ds = tf.contrib.distributions


class ProbabilisticEncoder(BaseNetwork):
    """Abstract base class for VAE encoder.


    """
    __metaclass__ = ABCMeta

    def __init__(self, params):
        super(ProbabilisticEncoder, self).__init__('Encoder', params)

    @property
    def latent_space_dim(self):
        return self._params['latent_space_dim']

    @abstractmethod
    def q_z_given_x(self, x, is_training):
        pass


class GaussianEncoder(ProbabilisticEncoder):
    """Base class for multivariate gaussian encoder"""

    def __init__(self, params):
        super(GaussianEncoder, self).__init__(params)

    def q_z_given_x(self, x, is_training):
        """
        Compute Q(Z|X)
        Args:
            x:
            is_training:

        Returns:
            An instance of ds.MultivariateNormalDiag distribution
        """
        means, log_sigmas = self.forward(x, is_training)
        return ds.MultivariateNormalDiag(means, log_sigmas)


class ProbabilisticDecoder(object):
    __metaclass__ = ABCMeta

    def __init__(self, params):
        self._params = params
        self._scope = 'Decoder'
        self._normalizer_fn = None # slim.batch_norm
        # self._normalizer_fn_args = {
        #     'is_training': is_training,
        #     'zero_debias_moving_mean': True,
        #     'fused': True,
        # }
        # self._p_x_given_z = self._compute_prob_x_given_z(latent_tensors, params)

    @abstractmethod
    def p_x_given_z(self, z, is_training, reuse=False):
        pass

    # @property
    # def mean(self):
    #     return self.p_x_given_z.mean()
    #
    # def get_variables(self):
    #     return tf.get_collection(tf.GraphKeys.VARIABLES, self._scope)
    #
    # def get_trainable_variables(self):
    #     return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)
    #
    # @abstractmethod
    # def _compute_prob_x_given_z(self, latent_tensors, params):
    #     pass
    #
    # @abstractmethod
    # def reconstruction_loss(self, targets):
    #     pass


class GaussianDecoder(ProbabilisticDecoder):
    """
    Base class for multivariate gaussian decoder.
    """

    def __init__(self, params):
        super(GaussianDecoder, self).__init__(params)

    def p_x_given_z(self, z, is_training, reuse=False):
        means, log_sigmas = self._mean_and_std(z, is_training, reuse)
        return ds.MultivariateNormalDiag(means, log_sigmas)

    @abstractmethod
    def _mean_and_std(self, z, is_training, reuse):
        """
        Compute mean and standard deviation.
        Args:
            z:
            is_training:

        Returns:
            (mean, std) tuple
        """
        return None, None
