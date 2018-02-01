from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

slim = tf.contrib.slim


class Encoder(object):
    """Abstract base class for probabilistic encoder model.


    """
    __metaclass__ = ABCMeta

    def __init__(self, obs_tensors, params, is_training):
        self._scope = 'Encoder'
        self._normalizer_fn = slim.batch_norm
        self._normalizer_fn_args = {
            'is_training': is_training,
            'zero_debias_moving_mean': True,
            'fused': True,
        }
        self._q_z_given_x = self._compute_prob_z_given_x(obs_tensors, params)

    @property
    def q_z_given_x(self):
        return self._q_z_given_x

    @abstractmethod
    def _compute_prob_z_given_x(self, obs_tensors, params):
        pass

