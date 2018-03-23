"""VAE experiments."""
import tensorflow as tf

from xae.estimators.vae import VAE
from xae.factories import vae_factory
from xae.parameters import vae_hparams
from xae.data import celeba_dataset

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    params = vae_hparams.default_params()

    config = tf.estimator.RunConfig(save_summary_steps=10)

    model_dir = '/tmp/vae'

    encoder = vae_factory.make_encoder(params.encoder)

    decoder = vae_factory.make_decoder(params.decoder)

    prior = vae_factory.make_prior(params.encoder['latent_space_dim'])

    estimator = VAE(encoder, decoder, prior, model_dir, params, config)

    estimator.train(lambda: celeba_dataset.image_file_inputs(params.train_data))

