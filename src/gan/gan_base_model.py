import tensorflow as tf
from absl import flags
from tensorflow.python import ops, math_ops
from gan.documentation import add_gan_scalars, get_gan_vars
from tensorflow.python.training.session_run_hook import SessionRunHook


class GAN(object):
    def __init__(self, data_handler):
        self.config = self.init_param()
        self.data_handler = data_handler
        self.dataset = self.config.dataset
        self.z_dim = self.config.z_dim
        self.gf_dim = self.config.gf_dim
        self.df_dim = self.config.df_dim
        self.global_step = tf.train.create_global_step()

        self.build_model()

    def init_param(self):
        return flags.FLAGS

    def build_model(self):
        """Builds a model."""
        config = self.config

        self.d_learning_rate, self.g_learning_rate = self.get_learning_rates()

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as model_scope:
            self.build_model_single_gpu(batch_size=config.batch_size)
            self.d_optim, self.g_optim, self.d_learning_rate, self.g_learning_rate = self.get_optimizers()
            # Add summaries.
            if self.config.running_mode == "train":
                add_gan_scalars(self.d_learning_rate, self.g_learning_rate, self.d_loss, self.d_loss_fake,
                                self.d_loss_real, self.g_loss, self.discriminator_real, self.discriminator_fake)

        self.add_trainable_parameters_to_tensorboard("Discriminator")
        self.add_trainable_parameters_to_tensorboard("Generator")

        self.add_gradients_to_tensorboard("Discriminator")
        self.add_gradients_to_tensorboard("Generator")

    def add_gradients_to_tensorboard(self, scope):
        [tf.summary.histogram(x.name.replace("model/", ""), x, family="Gradients_{}".format(scope))
         for x in tf.global_variables()
         if x not in tf.trainable_variables() and scope in x.name and ("g_opt" in x.name or "d_opt" in x.name)]

    def add_trainable_parameters_to_tensorboard(self, scope):
        [tf.summary.histogram(x.name.replace("model/", ""), x, family="Weights_{}".format(scope))
         for x in tf.trainable_variables() if scope in x.name and "beta" not in x.name and "gamma" not in x.name]

    def build_model_single_gpu(self, batch_size=1):
        config = self.config
        show_num = min(config.batch_size, 16)

        self.increment_global_step = self.global_step.assign_add(1)
        batch = self.data_handler.get_batch(batch_size, config)
        real_x, labels = batch[0], batch[1]

        real_x, labels = self.data_handler.prepare_real_data(real_x, labels)

        noise = tf.random_normal([config.batch_size, config.z_dim], dtype=tf.float32, name='z0')
        fake_x = self.get_generated_data(noise, labels)

        # real_x_mixed, fake_x_mixed, labels_mixed = self.random_shuffle(real_x, fake_x, labels, self.config.batch_size,
        #                                                                self.config.label_noise_level)
        self.discriminator_real, r_h = self.get_discriminator_result(real_x, labels)
        self.discriminator_fake, f_h = self.get_discriminator_result(fake_x, labels, reuse=True)
        self.discriminator_fake = tf.identity(self.discriminator_fake, name="d_score")
        self.data_handler.display_real_data(batch[0], labels, show_num, self.discriminator_real)
        self.data_handler.display_fake_data(fake_x, labels, show_num, self.discriminator_fake,
                                            self.config.running_mode == "train")

        # Loss
        self.d_loss_real, self.d_loss_fake, self.d_loss, self.g_loss = self.get_loss(real_x, fake_x, labels,
                                                                                     self.discriminator_real,
                                                                                     self.discriminator_fake, r_h, f_h)
        add_std_var_to_trensorboard(real_x, "Real")
        add_std_var_to_trensorboard(fake_x, "Fake")

        _, self.d_vars, self.g_vars = get_gan_vars()

    def get_optimizers(self):

        d_optimizer = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, name='d_opt', beta1=self.config.beta1,
                                             beta2=self.config.beta2)
        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, name='g_opt', beta1=self.config.beta1,
                                             beta2=self.config.beta2)
        g_optim = g_optimizer.minimize(self.g_loss, var_list=self.g_vars)
        beta1_power, beta2_power = d_optimizer._get_beta_accumulators()
        d_lr = (d_optimizer._lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        beta1_power, beta2_power = g_optimizer._get_beta_accumulators()
        g_lr = (g_optimizer._lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        return d_optim, g_optim, d_lr, g_lr

    def get_learning_rates(self):
        with tf.variable_scope('learning_rate'):
            # current_step = tf.cast(self.global_step, tf.float32)
            # g_ratio = (1.0 + 2e-5 * tf.maximum((current_step - 100000.0), 0.0))
            # g_ratio = tf.minimum(g_ratio, 4.0)

            return self.config.discriminator_learning_rate, self.config.generator_learning_rate

    def get_loss(self, discriminator_fake, discriminator_real, fake_x, gen_sparse_class, real_x, r_h, f_h):
        pass

    def get_discriminator_result(self, data, labels, reuse=False):
        discriminator_fn = self.get_discriminator_and_generator()[0]
        return discriminator_fn(data, labels, self.df_dim, self.data_handler.num_classes,
                                kernel=self.data_handler.get_kernel(),
                                strides=self.data_handler.get_strides(), dilations=self.data_handler.get_dilations(),
                                pooling=self.config.pooling, update_collection=None, reuse=reuse)

    def get_generated_data(self, data, labels):
        generator_fn = self.get_discriminator_and_generator()[1]
        return generator_fn(data, labels, self.gf_dim, self.data_handler.num_classes,
                            kernel=self.data_handler.get_kernel(),
                            strides=self.data_handler.get_strides(), dilations=self.data_handler.get_dilations(),
                            pooling=self.config.pooling, output_shape=self.data_handler.shape)

    def get_discriminator_and_generator(self):
        pass

    def random_shuffle(self, real, fake, labels, batch_size, label_noise_level):
        if label_noise_level > 0:
            perecentage_of_noise_data = tf.truncated_normal([1], mean=label_noise_level,
                                                            stddev=label_noise_level / 2.0)
            current_step = tf.cast(self.global_step, tf.float32)
            decay_factor = (2e-6 * tf.maximum((500000.0 - current_step), 0.0))
            num_of_not_swapped_examples = batch_size * (1 - perecentage_of_noise_data * decay_factor)
            idx = tf.random_shuffle(tf.range(int(batch_size)))
            num_real = tf.cast(tf.squeeze(num_of_not_swapped_examples), tf.int32)
            real_idx = tf.gather(idx, tf.range(num_real))
            fake_idx = tf.gather(idx, tf.range(num_real, int(batch_size)))

            real_ = tf.gather(real, real_idx)
            fake_ = tf.gather(fake, fake_idx)
            real_mix = tf.concat([real_, fake_], axis=0)

            fake_ = tf.gather(fake, real_idx)
            real_ = tf.gather(real, fake_idx)
            fake_mix = tf.concat([fake_, real_], axis=0)

            labels1 = tf.gather(labels, real_idx)
            labels2 = tf.gather(labels, fake_idx)
            labels_mix = tf.concat([labels1, labels2], axis=0)
            real_mix.set_shape([batch_size, *real_mix.get_shape().as_list()[1:]])
            fake_mix.set_shape([batch_size, *fake_mix.get_shape().as_list()[1:]])
            labels_mix.set_shape([batch_size, *labels_mix.get_shape().as_list()[1:]])
            return real_mix, fake_mix, labels_mix
        else:
            return real, fake, labels


def add_std_var_to_trensorboard(data, name):
    std = tf.reduce_mean(tf.keras.backend.std(data, axis=0))
    std = tf.Print(std, [std], "{} Stddev: ".format(name))
    tf.summary.scalar(name, std, family="Stddev")


class VariableRestorer(SessionRunHook):
    """Hook that counts steps per second."""

    def __init__(self, model_dir, variable_name_to_restore, variable_name_from_which_to_restore):
        self.model_dir = model_dir
        self.variable_name_to_restore = variable_name_to_restore
        self.variable_name_from_which_to_restore = variable_name_from_which_to_restore

    def begin(self):
        graph = ops.get_default_graph()
        self.acid_embeddings = graph.get_tensor_by_name("model/" + self.variable_name_to_restore + ":0")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        run_context.session.graph._unsafe_unfinalize()
        print("Restoring weights {} from model {} to variable {}".format(self.variable_name_from_which_to_restore,
                                                                         self.model_dir,
                                                                         self.variable_name_to_restore))
        saver_restore = tf.train.Saver({self.variable_name_from_which_to_restore: self.acid_embeddings})
        saver_restore.restore(run_context.session, self.model_dir)
        run_context.session.graph.finalize()
