"""The SNGAN Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gan.gan_base_model import GAN

from gan.sngan.discriminator import *
from gan.sngan.generator import *

tfgan = tf.contrib.gan


class SNGAN(GAN):
    """SNGAN model."""

    def __init__(self, data_handler):
        self.d_losses = tf.Variable(1.0, trainable=False)
        super(SNGAN, self).__init__(data_handler)

    def minibatch_std(self, x, function=tf.reduce_mean):
        y = x
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)  # Calc variance over group.
        y = tf.sqrt(y + 1e-8)  # Calc stddev over group.
        y = function(y)  # Take average of everything
        return y

    def calculate_difference(self, x, function=tf.reduce_mean):
        splitted = tf.split(x, 2)
        diff = tf.abs(splitted[0] - splitted[1])
        return function(diff)

    def get_loss(self, real_x, fake_x, gen_sparse_class, discriminator_real, discriminator_fake, r_h, f_h):
        if self.config.loss_type == 'hinge_loss_std':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_loss(fake_x, real_x)
            g_loss_gan = g_loss_gan + variation_loss
            print('hinge loss (+ std mean) is using')
        elif self.config.loss_type == 'hinge_loss_std_sum':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_loss(fake_x, real_x, tf.reduce_sum)
            g_loss_gan = g_loss_gan + variation_loss
            print('hinge loss ( + std sum ) is using')
        elif self.config.loss_type == 'hinge_loss':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_loss(discriminator_fake, discriminator_real)
            print('hinge loss is using')
        elif self.config.loss_type == 'hinge_loss_ra':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_ra_loss(discriminator_fake, discriminator_real)
            print('Relativistic hinge (std) loss is using')
        elif self.config.loss_type == 'hinge_loss_ra_std':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_ra_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_loss(fake_x, real_x)
            g_loss_gan = g_loss_gan + variation_loss
            print('Relativistic hinge loss is using')
        elif self.config.loss_type == 'hinge_loss_ra_std_sum':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_ra_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_loss(fake_x, real_x, tf.reduce_sum)
            g_loss_gan = g_loss_gan + variation_loss
            print('Relativistic hinge loss is using')
        elif self.config.loss_type == 'hinge_loss_diff':
            d_loss_fake, d_loss_real, g_loss_gan = self.hinge_loss(discriminator_fake, discriminator_real)
            variation_loss = self.get_variation_diff_loss(fake_x, real_x, tf.reduce_mean)
            g_loss_gan = g_loss_gan + self.config.variation_level * variation_loss
            print('Relativistic hinge loss is using')
        elif self.config.loss_type == 'kl_loss':
            d_loss_real = tf.reduce_mean(tf.nn.softplus(-discriminator_real))
            d_loss_fake = tf.reduce_mean(tf.nn.softplus(discriminator_fake))
            g_loss_gan = tf.reduce_mean(-discriminator_fake)
            print('kl loss is using')
        elif self.config.loss_type == 'wasserstein':
            d_loss_real = tf.reduce_mean(-discriminator_real)
            d_loss_fake = tf.reduce_mean(discriminator_fake)
            g_loss_gan = tf.reduce_mean(-discriminator_fake)
            print('wasserstein loss is using')
        else:
            raise NotImplementedError
        d_loss = d_loss_real + d_loss_fake
        g_loss = g_loss_gan
        return d_loss_real, d_loss_fake, d_loss, g_loss

    def hinge_ra_loss(self, discriminator_fake, discriminator_real):
        # Reference: https://github.com/taki0112/RelativisticGAN-Tensorflow/blob/master/ops.py
        fake_logit = (discriminator_fake - tf.reduce_mean(discriminator_real))
        real_logit = (discriminator_real - tf.reduce_mean(discriminator_fake))
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))
        g_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 - fake_logit))
        g_loss_real = tf.reduce_mean(tf.nn.relu(1.0 + real_logit))
        g_loss_gan = g_loss_fake + g_loss_real
        return d_loss_fake, d_loss_real, g_loss_gan

    def get_variation_diff_loss(self, fake_x, real_x, function=tf.reduce_mean):
        real_variance = self.calculate_difference(real_x, function=function)
        fake_variance = self.calculate_difference(fake_x, function=function)
        minibatch_variance = real_variance - fake_variance
        variation_loss = tf.maximum(0.0, minibatch_variance)
        # variation_loss = tf.Print(variation_loss, [variation_loss], "variation_loss", summarize=128)
        return variation_loss

    def get_variation_loss(self, fake_x, real_x, function=tf.reduce_mean):
        real_variance = self.minibatch_std(real_x, function=function)
        fake_variance = self.minibatch_std(fake_x, function=function)
        minibatch_variance = real_variance - fake_variance
        variation_loss = tf.maximum(0.0, minibatch_variance)
        variation_loss = tf.Print(variation_loss, [variation_loss], "variation_diff_loss", summarize=128)
        return variation_loss

    def hinge_loss(self, discriminator_fake, discriminator_real):
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - discriminator_real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + discriminator_fake))
        g_loss_gan = -tf.reduce_mean(discriminator_fake)

        return d_loss_fake, d_loss_real, g_loss_gan

    def get_discriminator_and_generator(self):

        if self.config.architecture == 'resnet':
            discriminator_fn = discriminator_resnet
            generator_fn = generator_resnet
        elif self.config.architecture == 'conditional':
            discriminator_fn = discriminator_conditional
            generator_fn = generator_conditional
        elif self.config.architecture == '1d':
            discriminator_fn = discriminator_1d
            generator_fn = generator_1d
        elif self.config.architecture == 'rnn':
            discriminator_fn = discriminator_rnn
            generator_fn = generator_rnn
        else:
            raise NotImplementedError

        return discriminator_fn, generator_fn
