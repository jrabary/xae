import tensorflow as tf
from math import log
from nets import dcgan

slim = tf.contrib.slim
ds = tf.contrib.distributions


def encode(x, latent_space_dim, is_training=True, scope='Encoder', fused_batch_norm=False):
    normalizer_fn = slim.batch_norm
    normalizer_fn_args = {
        'is_training': is_training,
        'zero_debias_moving_mean': True,
        'fused': fused_batch_norm,
    }

    depth = 128
    inp_shape = x.get_shape().as_list()[1]

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
                                                  activation_fn=tf.nn.softplus)

    q_z_given_x = ds.MultivariateNormalDiag(means, log_sigmas)

    return q_z_given_x


def decode(inputs, final_size, depth=64, is_training=True, num_outputs=3, fused_batch_norm=False, scope='Generator'):
    """Use dcgan generator architecture as decoder"""
    normalizer_fn = slim.batch_norm
    normalizer_fn_args = {
        'is_training': is_training,
        'zero_debias_moving_mean': True,
        'fused': fused_batch_norm,
    }

    inputs.get_shape().assert_has_rank(2)
    if log(final_size, 2) != int(log(final_size, 2)):
        raise ValueError('`final_size` (%i) must be a power of 2.' % final_size)
    if final_size < 8:
        raise ValueError('`final_size` (%i) must be greater than 8.' % final_size)

    end_points = {}
    num_layers = int(log(final_size, 2)) - 1
    with tf.variable_scope(scope, values=[inputs]):
        with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
            with slim.arg_scope([slim.conv2d_transpose],
                                normalizer_fn=normalizer_fn,
                                stride=2,
                                kernel_size=4):
                net = tf.expand_dims(tf.expand_dims(inputs, 1), 1)

                # First upscaling is different because it takes the input vector.
                current_depth = depth * 2 ** (num_layers - 1)
                scope = 'deconv1'
                net = slim.conv2d_transpose(
                    net, current_depth, stride=1, padding='VALID', scope=scope)
                end_points[scope] = net

                for i in range(2, num_layers):
                    scope = 'deconv%i' % (i)
                    current_depth = depth * 2 ** (num_layers - i)
                    net = slim.conv2d_transpose(net, current_depth, scope=scope)
                    end_points[scope] = net

                # Last layer has different normalizer and activation.
                scope = 'deconv%i' % (num_layers)
                net = slim.conv2d_transpose(
                    net, depth, normalizer_fn=None, activation_fn=None, scope=scope)
                end_points[scope] = net

                # Convert to proper channels.
                scope = 'means'
                means = slim.conv2d(
                    net,
                    num_outputs,
                    normalizer_fn=None,
                    activation_fn=None,
                    kernel_size=1,
                    stride=1,
                    padding='VALID',
                    scope=scope)
                end_points[scope] = means

                means.get_shape().assert_has_rank(4)
                means.get_shape().assert_is_compatible_with(
                    [None, final_size, final_size, num_outputs])

                scope = 'log_sigmas'
                log_sigmas = slim.conv2d(
                    net,
                    num_outputs,
                    normalizer_fn=None,
                    activation_fn=tf.nn.softplus,
                    kernel_size=1,
                    stride=1,
                    padding='VALID',
                    scope=scope)
                end_points[scope] = log_sigmas

                log_sigmas.get_shape().assert_has_rank(4)
                log_sigmas.get_shape().assert_is_compatible_with(
                    [None, final_size, final_size, num_outputs])

    p_x_given_z = ds.MultivariateNormalDiag(tf.reshape(means, [-1, final_size*final_size*num_outputs]),
                                            tf.reshape(log_sigmas, [-1, final_size*final_size*num_outputs]))
    return p_x_given_z
