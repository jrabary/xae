import tensorflow as tf

from abc import ABCMeta
from abc import abstractmethod

slim = tf.contrib.slim


class BaseNetwork(object):
    """
    Abstract base class for feed forward neural network.
    """
    __metaclass__ = ABCMeta

    def __init__(self, scope, params):
        self._scope = scope
        self._params = params

    def arg_scope(self, is_training):

        batch_norm = None
        batch_norm_params = None
        if 'batch_norm' in self._params:
            batch_norm = slim.batch_norm
            batch_norm_params = self._make_batch_norm_params(self._params['batch_norm'], is_training)

        affected_ops = [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose, slim.fully_connected]
        with slim.arg_scope(
                affected_ops,
                normalizer_fn=batch_norm,
                normalizer_params=batch_norm_params) as sc:
            return sc

    @abstractmethod
    def forward(self, inputs, is_training):
        pass

    def _make_batch_norm_params(self, params, is_training):
        """
        Build the batch norm parameters
        Args:
            params: default batch norm parameters
            is_training: if train or test mode

        Returns:

        """

        batch_norm_params = params

        if is_training:
            batch_norm_params['is_training'] = True

        return batch_norm_params
