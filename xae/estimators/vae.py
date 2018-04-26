import tensorflow as tf
import tensorflow_probability as tfp

ds = tf.contrib.distributions
slim = tf.contrib.slim
tfgan = tf.contrib.gan


class VAE(tf.estimator.Estimator):
    """
    Variational Auto-Encoder estimator definition.
    """

    def __init__(self, encoder, decoder, prior, model_dir, params, run_config):
        """
        Define VAE estimator
        Args:
            encoder: an instance of VAEEncoder
            decoder: an instance of VEADecoder
            prior:
        """
        self._model_dir = model_dir
        self._params = params
        self._run_config = run_config
        self._encoder = encoder
        self._decoder = decoder
        self._prior = prior

        super(VAE, self).__init__(self._model_fn, model_dir, run_config, params)

    def _model_fn(self, features, labels, mode, params):
        """
        VAE estimator model function
        Args:
            features: real images
            labels: real images
            mode: estimator run mode
            params: parameters

        Returns:
            EstimatorSpec to train VAE.

        """

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        x = features

        # Get the prior P(Z)
        prior = self._prior()

        # Get the encoder probability distribution: Q(Z|X)
        q_z_given_x = self._encoder.q_z_given_x(x, is_training)

        # Compute log P(X, Z)
        def joint_log_prob(z):
            # Get the decoder probability distribution: Pg(X|Z)
            p_x_given_z = self._decoder.p_x_given_z(z, is_training)
            return p_x_given_z.log_prob(tf.layers.flatten(x)) + prior.log_prob(z)

        # Compute the Evidence Lower Bound Loss using csiszar_divergence in tfp
        # ELBO(x) = D_f(P(X=x, Z), Q(Z|X=x) with f = -log
        #         = log(P(X=x|Z)) - KL(Q(Z|X=x, P(Z)) another way to see the ELBO
        elbo_loss = tf.reduce_sum(
            tfp.vi.csiszar_divergence.monte_carlo_csiszar_f_divergence(
                f=tfp.vi.csiszar_divergence.kl_reverse,
                p_log_prob=joint_log_prob,
                q=q_z_given_x,
                num_draws=1))

        tf.summary.scalar('elbo', elbo_loss)

        output_shape = [params.batch_size] + 2*[params.decoder['output_size']] + [params.decoder['output_channels']]

        if mode == tf.estimator.ModeKeys.TRAIN:

            # Visualize training data reconstruction
            # Display the mean of P(X|Z=encoded_x)
            encoded_x = q_z_given_x.sample(1)
            p_x_given_encoded_x = self._decoder.p_x_given_z(encoded_x, is_training, reuse=True)
            x_mean = tf.reshape(p_x_given_encoded_x.mean(), output_shape)
            x_mean.set_shape(output_shape)
            reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
            features.set_shape([params.batch_size] + params.input_shape)
            targets = tfgan.eval.image_reshaper(features, num_cols=8)
            tf.summary.image('Reconstruction/x_mean', reconstruction)
            tf.summary.image('Reconstruction/x_targets', targets)

            # Visualize random samples
            noise_samples = tf.expand_dims(prior.sample(params.batch_size), axis=0)
            p_x_given_noise = self._decoder.p_x_given_z(noise_samples, is_training, reuse=True)
            x_mean = tf.reshape(p_x_given_noise.mean(), output_shape)
            x_mean.set_shape(output_shape)
            reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
            tf.summary.image('Sample/x_mean', reconstruction)

            # Optimize
            optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
            train_op = optimizer.minimize(elbo_loss, tf.train.get_or_create_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=elbo_loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:

            noise_samples = tf.expand_dims(prior.sample(params.batch_size), axis=0)
            p_x_given_noise = self._decoder.p_x_given_z(noise_samples, is_training, reuse=True)
            x_mean = tf.reshape(p_x_given_noise.mean(), output_shape)
            x_mean.set_shape(output_shape)
            reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
            tf.summary.image('Reconstruction/x_mean', reconstruction)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=elbo_loss
            )
