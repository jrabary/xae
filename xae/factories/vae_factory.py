import tensorflow as tf

# encoders import
from xae.nets.vae_dcgan_encoder import VAEDCGanEncoder

# decoders import
from xae.nets.vae_dcgan_decoder import VAEDCGanDecoder


ds = tf.contrib.distributions


_VAE_ENCODERS = {
    'dcgan': VAEDCGanEncoder
}

_VAE_DECODERS = {
    'dcgan': VAEDCGanDecoder
}


def make_encoder(params):
    """ VAE encoder factory. """
    if params['name'] not in _VAE_ENCODERS:
        raise ValueError('Unknown encoder: {}'.format(params['name']))

    return _VAE_ENCODERS[params['name']](params)


def make_decoder(params):
    """ VAE decoder factory. """
    if params['name'] not in _VAE_DECODERS:
        raise ValueError('Unknown decoder: {}'.format(params['name']))

    return _VAE_DECODERS[params['name']](params)


def make_prior(latent_space_dim):

    def prior_fn():
        return ds.MultivariateNormalDiag(loc=tf.zeros([1, latent_space_dim], dtype=tf.float32),
                                         scale_diag=tf.ones([1, latent_space_dim], dtype=tf.float32))

    return prior_fn
