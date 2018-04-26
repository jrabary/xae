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
        return ds.MultivariateNormalDiag(scale_diag=tf.ones(latent_space_dim, dtype=tf.float32), name='Prior')

    return prior_fn
