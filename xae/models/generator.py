from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

slim = tf.contrib.slim


class Generator(object):
    """Abstract base class for probabilistic generator model.


    """
    __metaclass__ = ABCMeta

    def __init__(self, latent_tensors, params, is_training):
        self._scope = 'Generator'
        self._normalizer_fn = slim.batch_norm
        self._normalizer_fn_args = {
            'is_training': is_training,
            'zero_debias_moving_mean': True,
            'fused': True,
        }
        self._p_x_given_z = self._compute_prob_x_given_z(latent_tensors, params)

    @property
    def p_x_given_z(self):
        return self._p_x_given_z

    @property
    def mean(self):
        return self.p_x_given_z.mean()

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self._scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)

    @abstractmethod
    def _compute_prob_x_given_z(self, latent_tensors, params):
        pass

    @abstractmethod
    def reconstruction_loss(self, targets):
        pass


