import os

import common.model.utils_ori as utils
import numpy as np
import tensorflow as tf
from bio.amino_acid import protein_seq_to_string

from bio.constants import ID_TO_AMINO_ACID, get_lesk_color_mapping
from gan.documentation import add_image_grid
from gan.protein.helpers import convert_to_acid_ids, REAL_PROTEINS, ACID_EMBEDDINGS_SCOPE, ACID_EMBEDDINGS, \
    FAKE_PROTEINS
from model import ops
from model.ops import pad_up_to

LABELS = "labels"



class Protein(object):

    def __init__(self, flags, properties):
        self.config = flags
        self.properties = properties
        self.num_classes = len(properties["class_mapping"])
        self.width = properties["seq_length"]
        self.embeddings = np.load(
            os.path.join(flags.data_dir, *["protein", "hand_crafted_embeddings_{}.npy".format(self.config.embedding_height)]))
        self.embeddings_variation = np.load(
            os.path.join(flags.data_dir, *["protein", "embeddings_variation_{}.npy".format(self.config.embedding_height)]))
        self.reactions = np.load(
            os.path.join(flags.data_dir, flags.dataset.replace("\\", os.sep), flags.running_mode + "_reactions.npy"))
        self.acid_embeddings = None
        self.shape = [self.config.embedding_height, properties["seq_length"], 1]

    def get_kernel(self):
        if self.config.one_hot:
            return 21, 3
        else:
            return self.config.kernel_height, self.config.kernel_width

    def get_strides(self):
        if self.width > 256:
            strides_schedule = [(1, 2), (1, 2), (2, 2), (2, 2), (2, 2)]
        else:
            strides_schedule = [(1, 2), (2, 2), (2, 2), (2, 2)]
        return strides_schedule

    def get_dilations(self):
        return 1, self.config.dilation_rate

    def prepare_real_data(self, real_x, labels):
        real_x = tf.identity(real_x, name=REAL_PROTEINS)
        reactions = self.get_reactions(labels)
        labels = tf.identity(labels, name=LABELS)
        if self.config.one_hot:
            real_x = self.get_real_data_to_display(real_x)
        else:
            real_x = self.get_embedded_seqs(real_x)
        return real_x, (labels, reactions)

    def display_real_data(self, real_x, labels, show_num, d_scores=None):
        if self.config.running_mode == "train":
            self.display_metrics_about_protein(real_x, "real")
            real_to_display = self.get_real_data_to_display(real_x)
            self.display_protein(real_to_display, labels[0], show_num, "real", self.width, d_scores)

    def get_real_data_to_display(self, real_x):
        real_to_display = tf.one_hot(real_x, 21, axis=1)
        real_to_display = tf.expand_dims(real_to_display, 3)
        return real_to_display

    def get_embedded_seqs(self, real_x):
        if self.config.static_embedding:
            if self.acid_embeddings is None:
                with tf.variable_scope(ACID_EMBEDDINGS_SCOPE):
                    self.acid_embeddings = tf.get_variable(ACID_EMBEDDINGS,
                                                           shape=[len(ID_TO_AMINO_ACID), self.config.embedding_height],
                                                           initializer=tf.constant_initializer(self.embeddings),
                                                           trainable=False)
                real_x_emb = tf.nn.embedding_lookup(self.acid_embeddings, real_x)
            real_x_emb = self.add_noise(real_x, real_x_emb)
        else:
            real_x_emb = ops.sn_embedding(real_x, len(ID_TO_AMINO_ACID), self.config.embedding_height, name=ACID_EMBEDDINGS_SCOPE,
                                          embedding_map_name=ACID_EMBEDDINGS)
            real_x_emb.set_shape([self.config.batch_size, self.properties["seq_length"], self.config.embedding_height])
        real_x_emb = tf.transpose(real_x_emb, perm=[0, 2, 1])
        real_x_emb = tf.expand_dims(real_x_emb, 3)
        return real_x_emb

    def add_noise(self, real_x, real_x_emb):
        real_x_one_hot = tf.one_hot(real_x, 21, axis=1)

        variation = []
        for embbedding in self.embeddings_variation:
            embbedding_variation = []
            for acid in embbedding:
                # embbedding_variation.append(tf.random_uniform([1], minval=acid[0], maxval=acid[1]))
                embbedding_variation.append(
                    tf.truncated_normal([self.config.batch_size], mean=(acid[0] + acid[1]) / 2.0,
                                        stddev=abs(acid[0] - acid[1]) / 4.0))
            variation.append(embbedding_variation)

        t_variation = tf.squeeze(tf.convert_to_tensor(variation))
        t_variation = tf.transpose(t_variation, perm=[2, 0, 1])
        t_variation = self.config.noise_level * t_variation
        noise = tf.matmul(t_variation, real_x_one_hot)
        noise = tf.transpose(noise, perm=[0, 2, 1])
        # noise_extra = tf.random_normal([self.config.batch_size, self.width, self.config.input_h],
        #                          stddev=0.001, dtype=tf.float32)
        noise_extra = tf.random_uniform([self.config.batch_size, self.width, self.config.embedding_height],
                                        maxval=0.001, minval=-0.001, dtype=tf.float32)
        real_x_emb = tf.add(real_x_emb, noise) + noise_extra
        return real_x_emb

    def display_fake_data(self, fake_x, labels, show_num, d_scores= None, to_display=True):
        if self.config.one_hot:
            fake_to_display = tf.argmax(fake_x, axis=1)
            fake_to_display = tf.squeeze(fake_to_display, name=FAKE_PROTEINS)
        else:
            fake_to_display = convert_to_acid_ids(fake_x, self.config.batch_size)
            fake_to_display = tf.Print(fake_to_display, [fake_to_display[0]], "FAKE", summarize=self.width)
        if to_display:
            self.display_metrics_about_protein(fake_to_display, "fake")
            fake_to_display = tf.one_hot(fake_to_display, 21, axis=1)
            self.display_protein(fake_to_display, labels[0], show_num, "fake", self.width, d_scores)

    def display_protein(self, protein, labels, show_num, family, protein_len, d_scores = None):
        protein_to_display = self.color_protein(protein, protein_len)
        image_shape = [protein_to_display.shape[1], protein_to_display.shape[2], protein_to_display.shape[3]]
        add_image_grid(family + "_image_grid", show_num, protein_to_display, image_shape, (show_num, 1))
        with tf.variable_scope(self.config.running_mode, reuse=True):
            tf.summary.text(family, tf.py_func(lambda vals, labels, d_scores: "\n".join(
                protein_seq_to_string(vals, self.properties["class_mapping"], labels, d_scores)),
                                               [tf.argmax(tf.squeeze(protein), axis=1), labels, d_scores], tf.string))

    def display_metrics_about_protein(self, data, family):
        flatten = tf.reshape(data, [-1])
        y, idx, count = tf.unique_with_counts(flatten)
        tf.summary.scalar(family, tf.size(count), family="unique_amino_acids")
        tf.summary.histogram(family, flatten, family="distribution_of_values")

    def color_protein(self, protein, protein_len=128):
        protein = tf.squeeze(protein)
        protein = tf.transpose(protein, perm=[0, 2, 1])
        colors = tf.expand_dims(get_lesk_color_mapping(), 0)
        colors = tf.tile(colors, [self.config.batch_size, 1, 10])
        colored = tf.matmul(protein, colors)
        colored = tf.reshape(colored, [colored.shape[0], protein_len, 10, 3])
        colored = tf.transpose(colored, perm=[0, 2, 1, 3])
        return colored

    def get_batch(self, batch_size, config):
        path = os.path.join(config.data_dir, config.dataset.replace("\\", os.sep))
        batches = utils.get_batches(utils.extract_seq_and_label, path, batch_size, cycle_length=self.num_classes,
                                    shuffle_buffer_size=config.shuffle_buffer_size, running_mode=config.running_mode,
                                    args=[[self.width], self.config.dynamic_padding])
        return batches

    def get_reactions(self, labels):
        reaction_tensors = []
        for reaction in self.reactions:
            # label = tf.convert_to_tensor(reaction[0])
            r = [pad_up_to(tf.convert_to_tensor(reaction[i]), [self.config.compound_w],  self.config.dynamic_padding) for i in range(1, 5)]
            reaction_tensors.append(r)
        reaction_tensors = tf.stack(reaction_tensors)
        return tf.gather_nd(reaction_tensors, tf.expand_dims(labels, axis=1))