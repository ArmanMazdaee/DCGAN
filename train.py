import os
import tensorflow as tf
import numpy as np
from cifar10_input import load_cifar10
import models


def train(dataset_paths, batch_size, num_steps, output_dir, save_per_steps):
    dataset = load_cifar10(dataset_paths, batch_size)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'images')
            noise = tf.placeholder(tf.float32, [None, 100], 'noise')

        generated_train = models.generator(
            noise=noise,
            training=True,
            reuse=False)
        generated_predict = models.generator(
            noise=noise,
            training=False,
            reuse=True)

        discriminated_real = models.discriminator(
            images=generated_predict,
            training=True,
            reuse=False)
        discriminated_fake = models.discriminator(
            images=images,
            training=True,
            reuse=True)
        discriminated_predict = models.discriminator(
            images=generated_train,
            training=False,
            reuse=True)

        generator_loss = tf.reduce_mean(
            -tf.log(discriminated_predict))
        discriminator_loss = tf.reduce_mean(
            -tf.log(discriminated_real) - tf.log(1-discriminated_fake))

        generator_optimizer = tf.train.AdamOptimizer()
        generator_trainable = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='generator')
        generator_train_op = generator_optimizer.minimize(
            loss=generator_loss,
            var_list=generator_trainable,
            name='generator_train_op')
        discriminator_optimizer = tf.train.AdamOptimizer()
        discriminator_trainable = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='discriminator')
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=discriminator_trainable,
            name='discriminator_train_op')

        generator_summaries = tf.summary.scalar(
            'generator_loss', generator_loss)
        discriminator_summaries = tf.summary.scalar(
            'discriminator_loss', discriminator_loss)
        summaries_writer = tf.summary.FileWriter(
            os.path.join(output_dir, 'logs/'))

        saver = tf.train.Saver(max_to_keep=None, save_relative_paths=True)
        saver_path = os.path.join(output_dir, 'checkpoints', 'step')

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for step in range(1, num_steps+1):
                print('step', step)
                noise_value = np.random.rand(batch_size, 100)
                _, summaries_value = sess.run(
                    fetches=[generator_train_op, generator_summaries],
                    feed_dict={noise: noise_value})
                summaries_writer.add_summary(summaries_value, step)

                noise_value = np.random.rand(batch_size, 100)
                images_value = next(dataset)
                _, summaries_value = sess.run(
                    fetches=[discriminator_train_op, discriminator_summaries],
                    feed_dict={noise: noise_value, images: images_value})
                summaries_writer.add_summary(summaries_value, step)

                if step % save_per_steps == 0:
                    saver.save(
                        sess=sess,
                        save_path=saver_path,
                        global_step=step)


def main():
    dataset_paths = ['cifar-10-batches-bin/data_batch_%d.bin' % i
                     for i in range(1, 6)]
    train(dataset_paths, 16, 50, 'output', 10)

if __name__ == '__main__':
    main()
