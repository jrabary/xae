import tensorflow as tf


def default_params():

    return tf.contrib.training.HParams(

        input_shape=[32, 32, 3],

        encoder={
            'name': 'dcgan',
            'latent_space_dim': 2,
            'depth': 64
        },

        decoder={
            'name': 'dcgan',
            'depth': 64,
            'output_size': 32,
            'output_channels': 3
        },

        train_data='/Users/jaonary/Data/celebA/img_align_celeba/*.jpg',

        batch_size=32,

        learning_rate=1e-3
    )
