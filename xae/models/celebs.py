import tensorflow as tf
from math import log
from nets import dcgan

slim = tf.contrib.slim


def encode(x, latent_space_dim, is_training=True, scope='Encoder', fused_batch_norm=False):
    normalizer_fn = slim.batch_norm
    normalizer_fn_args = {
        'is_training': is_training,
        'zero_debias_moving_mean': True,
        'fused': fused_batch_norm,
    }

    depth = 128
    inp_shape = x.get_shape().as_list()[1]

    end_points = {}
    with tf.variable_scope(scope, values=[x]):
        with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
            with slim.arg_scope([slim.conv2d],
                                stride=2,
                                kernel_size=4,
                                activation_fn=tf.nn.leaky_relu):
                net = x
                # for i in range(int(log(inp_shape, 2))):
                for i in range(4):
                    scope = 'conv%i' % (i + 1)
                    current_depth = depth * 2 ** i
                    normalizer_fn_ = None if i == 0 else normalizer_fn
                    net = slim.conv2d(
                        net, current_depth, normalizer_fn=normalizer_fn_, scope=scope)

                net = slim.flatten(net)

                means = slim.fully_connected(net,
                                             latent_space_dim,
                                             normalizer_fn=None,
                                             activation_fn=None)

                log_sigmas = slim.fully_connected(net,
                                                  latent_space_dim,
                                                  normalizer_fn=None,
                                                  activation_fn=None)

                return means, log_sigmas


def decode(z, observable_space_dim, is_training=True):
    """Use dcgan generator architecture as decoder"""
    logits, _ = dcgan.generator(z, final_size=observable_space_dim, is_training=is_training)
    return logits
