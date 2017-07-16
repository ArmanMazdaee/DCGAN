import tensorflow as tf


def generator(noise, training=False, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        with tf.variable_scope('projector'):
            x = tf.layers.dense(noise, units=1024 * 4 * 4)
            x = tf.reshape(x, [-1, 4, 4, 1024])
            x = tf.layers.batch_normalization(
                inputs=x,
                training=training)
            x = tf.nn.relu(x)

        filters_sizes = [512, 256]
        for i, filters in enumerate(filters_sizes):
            with tf.variable_scope('deconv_%d' % i):
                x = tf.layers.conv2d_transpose(
                    inputs=x,
                    filters=filters,
                    kernel_size=(5, 5),
                    strides=(2, 2),
                    padding='SAME')
                x = tf.layers.batch_normalization(
                    inputs=x,
                    training=training)
                x = tf.nn.relu(x)

        with tf.variable_scope('deconv_last'):
            x = tf.layers.conv2d_transpose(
                inputs=x,
                filters=3,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='SAME')
            x = tf.nn.tanh(x)
    return x


def discriminator(images, training=False, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = images
        filters_sizes = [512, 256]
        for i, filters in enumerate(filters_sizes):
            with tf.variable_scope('conv_%d' % i):
                x = tf.layers.conv2d(
                    inputs=x,
                    filters=filters,
                    kernel_size=(5, 5),
                    strides=(2, 2),
                    padding='SAME')
                x = tf.layers.batch_normalization(
                    inputs=x,
                    training=training)
                x = tf.nn.softplus(x)

        with tf.variable_scope('classifier'):
            size = int(x.shape[1] * x.shape[2] * x.shape[3])
            x = tf.reshape(x, [-1, size])
            x = tf.layers.dense(
                inputs=x,
                units=1)
            x = tf.nn.sigmoid(x)
    return x
