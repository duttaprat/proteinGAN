import tensorflow as tf

from pathlib import Path
from datetime import datetime

from gan.documentation import print_model_summary

import numpy as np


class auto_encoder:
    """Simple auto encoder (some ideas were borrowed from https://arxiv.org/pdf/1708.01715.pdf)
    Some findings:
        1) SELU performs better than other activation functions, elu, leaky_relu (accuracy: 0.914 vs 0.86))
        2) Batch Norm helps to achive higher accuracy (accuracy 0.92 vs 0.914)
        3) Xavier Initialization did not have any impact
        4) Simple auto encoder tends to overfit after second epoch - high dropout rate is needed
        5) Momentum Optimizer is very slow or needs high learning rate.
    """

    def __init__(self, sequence_length, size_one_hot, model_version="v1", model_path="../logs/auto_encoder/",
                 batch_size=128, dropout_rate=0.4, learning_rate=0.001, encoder_layers=None, decoder_layers=None,
                 min_acc_to_save=0.9):

        if decoder_layers is None:
            decoder_layers = []
        if encoder_layers is None:
            encoder_layers = [1024]
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.size_one_hot = size_one_hot
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.min_acc_to_save = min_acc_to_save
        self.model_path = model_path
        self.model_version = model_version
        self.init_model()

    def init_model(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = False

        self.graph = self.create_model()
        self.sess = tf.Session(graph=self.graph)

    def encoder(self, x, layers, is_training):
        with tf.variable_scope('encoder'):
            encoded = tf.layers.flatten(x, name="flat")
            i = 0
            for layer in layers:
                encoded = tf.layers.dense(inputs=encoded,
                                          activation=tf.nn.selu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=layer,
                                          name="dense{}".format(i))
                i = i + 1
            encoded = tf.layers.batch_normalization(encoded, name="batch_normalization_encoder")

        return encoded

    def decoder(self, x, layers, is_training):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            i = 0
            decoded = x
            for layer in reversed(layers):
                decoded = tf.layers.dense(inputs=decoded,
                                          activation=tf.nn.selu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=layer,
                                          name="dense{}".format(i))
                i = i + 1
            decoded = tf.layers.batch_normalization(decoded, name="batch_normalization_decoder")
            decoded = tf.layers.dense(inputs=decoded,
                                      activation=None,
                                      units=self.sequence_length * self.size_one_hot,
                                      name="final_dense")
            decoded = tf.reshape(decoded, shape=[-1, self.sequence_length, self.size_one_hot], name='decoded')
        return decoded

    def create_model(self):
        auto_encoder_graph = tf.Graph()
        with auto_encoder_graph.as_default():
            tf.set_random_seed(10)

            with tf.variable_scope('input'):
                self.sequences = tf.placeholder(tf.int32, [None, self.sequence_length], name='sequences')
                self.is_training = tf.placeholder(tf.bool, name='is_train')

            dataset = (tf.data.Dataset.from_tensor_slices(self.sequences)
                       .shuffle(buffer_size=10000, reshuffle_each_iteration=True)
                       .apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)))

            self.iterator = dataset.make_initializable_iterator()

            self.batch_sequences = self.iterator.get_next()

            with tf.variable_scope('one_hot'):
                self.one_hot_seq = tf.one_hot(self.batch_sequences, self.size_one_hot)

            self.encoded = self.encoder(self.one_hot_seq, self.encoder_layers, self.is_training)
            self.encoded = tf.layers.dropout(self.encoded, self.dropout_rate, name="dropout", training=self.is_training)
            self.decoded = self.decoder(self.encoded, self.decoder_layers, self.is_training)

            # Define loss and optimizer
            with tf.name_scope("loss_op"):
                self.loss_op = tf.losses.sparse_softmax_cross_entropy(self.batch_sequences, self.decoded)
                self.correct_prediction = tf.equal(tf.argmax(self.decoded, 2, output_type=tf.int32),
                                                   self.batch_sequences)
                self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                tf.summary.scalar("loss_op", self.loss_op)

            with tf.name_scope("optimizer"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss_op)

            self.summ = tf.summary.merge_all()

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            print_model_summary()
        return auto_encoder_graph

    def train(self, train_data, val_data, n_of_epochs=3):
        batches_per_epoch = int(train_data.shape[0] / self.batch_size) + 1
        log_dir = "{}{}".format(self.model_path, datetime.now().strftime("%Y%m%d_%H%M"))
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        tb_writer = tf.summary.FileWriter(log_dir, self.graph)
        # Run the initializer
        epoch, step = 0, 0
        self.sess.run([self.init, self.iterator.initializer], feed_dict={self.sequences: train_data,
                                                                         self.is_training: True})
        while epoch < n_of_epochs:
            try:
                self.sess.run(self.train_op, feed_dict={self.is_training: True})
                step = step + 1
                if step % int(batches_per_epoch / 4) == 0 or step == 1:
                    loss, a = self.sess.run([self.loss_op, self.acc], feed_dict={self.is_training: False})
                    self.print_progress(step, loss, a)
                    [s] = self.sess.run([self.summ], feed_dict={self.is_training: False})
                    # tb_writer.add_summary(np.mean(loss), step)
            except tf.errors.OutOfRangeError:
                epoch = epoch + 1
                val_acc = self.validate(self.sess, epoch, val_data)
                self.save(val_acc)
                self.sess.run(self.iterator.initializer, feed_dict={self.sequences: train_data,
                                                                    self.is_training: True})
        print("Optimization Finished!")

    def save(self, val_acc):
         if val_acc > self.min_acc_to_save:
            self.min_acc_to_save = val_acc
            save_path = self.saver.save(self.sess, "{}{}".format(self.model_path, self.model_version))
            print("Model saved in path: %s" % save_path)

    def predict(self, val_data):
        self.sess.run(self.iterator.initializer, feed_dict={self.sequences: val_data, self.is_training: False})
        decoded_to_index = tf.argmax(self.decoded, axis=2)
        original, decoded = self.sess.run([self.batch_sequences, decoded_to_index], feed_dict={self.is_training: False})
        return original, decoded

    def restore(self):
        saved_path = "{}{}".format(self.model_path, self.model_version)
        self.saver.restore(self.sess, saved_path)

    def print_progress(self, step, loss, acc):
        print("Step {}, Loss={:.4f}, Accuracy={:.3f}".format(str(step), loss, acc))

    def validate(self, sess, epoch, val_data):
        # Calculate batch loss and accuracy
        losses = []
        accuracies = []
        sess.run(self.iterator.initializer, feed_dict={self.sequences: val_data, self.is_training: False})
        while True:
            try:
                loss, a = sess.run([self.loss_op, self.acc], feed_dict={self.is_training: False})
                losses.append(loss)
                accuracies.append(a)
            except tf.errors.OutOfRangeError:
                break
        loss_avg = sum(losses) / len(losses)
        acc_avg = sum(accuracies) / len(accuracies)
        self.print_progress("VALIDATION for epoch {}".format(epoch), loss_avg, acc_avg)
        return acc_avg

    def encode(self, data):
        self.sess.run(self.iterator.initializer, feed_dict={self.sequences: data, self.is_training: False})
        encoded = None
        while True:
            try:
                _encoded = self.sess.run(self.encoded, feed_dict={self.is_training: False})
                if encoded is None:
                    encoded = _encoded
                else:
                    encoded = np.append(encoded, _encoded, axis=0)
            except tf.errors.OutOfRangeError:
                print("Finished encoding. Encoded shape: {}".format(encoded.shape))
                break;
        return encoded

    def decode(self, data):
        with self.graph.as_default():
            encoded = tf.placeholder(tf.float32, [None, self.encoder_layers[-1]], name='sequences')
            decoded_to_index = tf.argmax(self.decoder(encoded, self.decoder_layers, self.is_training), axis=2)
        decoded = self.sess.run(decoded_to_index, feed_dict={encoded: data, self.is_training: False})

        return decoded
