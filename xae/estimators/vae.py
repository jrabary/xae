import tensorflow as tf

ds = tf.contrib.distributions
slim = tf.contrib.slim
tfgan = tf.contrib.gan


class VAE(tf.estimator.Estimator):

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

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        x = features
        real_x = labels

        # Get the prior
        with tf.variable_scope('prior'):
            prior = self._prior()

        # Get the encoder probability distribution: Q(Z|X)
        q_z_given_x = self._encoder.q_z_given_x(x, is_training)

        z_samples = q_z_given_x.sample()

        # Get the decoder probability distribution: Pg(X|Z)
        p_x_given_z = self._decoder.p_x_given_z(z_samples, is_training)

        output_shape = [params.batch_size] + 2*[params.decoder['output_size']] + [params.decoder['output_channels']]

        # compute loss
        # KL can be seen as regularization term!
        KL = ds.kl_divergence(q_z_given_x, prior)

        # The ELBO = reconstruction term + regularization term
        reconstruction_loss = p_x_given_z.log_prob(slim.flatten(real_x))
        tf.summary.scalar('Losses/reconstruction', tf.reduce_mean(reconstruction_loss))
        tf.summary.scalar('Losses/kl', tf.reduce_mean(KL))

        elbo = tf.reduce_mean(reconstruction_loss - KL)
        loss = -elbo

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Display the mean of P(X|Z)
            x_mean = tf.reshape(p_x_given_z.mean(), output_shape)
            x_mean.set_shape(output_shape)
            reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
            features.set_shape([params.batch_size] + params.input_shape)
            targets = tfgan.eval.image_reshaper(features, num_cols=8)
            tf.summary.image('Reconstruction/x_mean', reconstruction)
            tf.summary.image('Reconstruction/x_targets', targets)

            noise_samples = prior.sample(params.batch_size)
            p_x_given_noise = self._decoder.p_x_given_z(noise_samples, is_training, reuse=True)
            x_mean = tf.reshape(p_x_given_noise.mean(), output_shape)
            x_mean.set_shape(output_shape)
            reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
            tf.summary.image('Sample/x_mean', reconstruction)

            optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
            train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:

            noise_samples = prior.sample(params.batch_size)
            p_x_given_noise = self._decoder.p_x_given_z(noise_samples, is_training, reuse=True)
            x_mean = tf.reshape(p_x_given_noise.mean(), output_shape)
            x_mean.set_shape(output_shape)
            reconstruction = tfgan.eval.image_reshaper(x_mean, num_cols=8)
            tf.summary.image('Reconstruction/x_mean', reconstruction)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss
            )
