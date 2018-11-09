"""Generate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf

from bio.amino_acid import print_protein_seq
from gan.models import get_model
from gan.parameters import get_flags
from gan.documentation import setup_logdir, get_properties
from tensorflow.contrib.training import evaluate_repeatedly
import numpy as np

slim = tf.contrib.slim
tfgan = tf.contrib.gan

FLAGS = get_flags()


def main(_):
    FLAGS.properties_file = "properties_test.json"
    properties = get_properties(FLAGS)
    logdir = setup_logdir(FLAGS, properties)
    FLAGS.running_mode = "test"
    path = os.path.join(FLAGS.data_dir, FLAGS.dataset.replace("\\", os.sep), "d_test.csv")
    # data_to_test = pd.read_csv(path, header=None)[:FLAGS.batch_size]
    # data_to_test = from_amino_acid_to_id(data_to_test, 0)

    model = get_model(FLAGS, properties)

    path = os.path.join(FLAGS.data_dir, FLAGS.dataset.replace("\\", os.sep), "properties_test_original.json")
    with open(path) as json_data_file:
        properties = json.load(json_data_file)



    with tf.variable_scope('model', reuse=True):
        real_x, labels = model.data_handler.get_batch(FLAGS.batch_size, FLAGS)
        real_x, labels = model.data_handler.prepare_real_data(real_x, labels)
        #
        # data = model.data_handler.get_embedded_seqs(data_to_test)
        # d_scores, _ = model.get_discriminator_result(data, batch[1:], reuse=True)
        # printing_d_scores = tf.py_func(
        #     lambda vals, scores: print_protein_seq(vals, properties["class_mapping"], d_scores=scores),
        #     [tf.squeeze(data_to_test), d_scores], tf.string)

        noise = np.random.normal(size=[FLAGS.batch_size, FLAGS.z_dim], loc=0.0, scale=1.0)
        z = noise
        generated_data = model.get_generated_data(tf.convert_to_tensor(z, dtype=tf.float32), labels)
        d_fake_scores, _ = model.get_discriminator_result(generated_data, labels, reuse=True)
        generated_data_ids = model.data_handler.convert_to_acid_ids(generated_data)
        printing_sequeces = tf.py_func(
            lambda vals, labels: print_protein_seq(vals, properties["class_mapping"], labels=labels),
            [tf.squeeze(generated_data_ids), labels[0]], tf.string)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    evaluate_repeatedly(
        checkpoint_dir=logdir,
        hooks=[
            # BlastHook(id_to_enzyme_class_dict, every_n_steps=1, output_dir=logdir, n_examples=FLAGS.batch_size,
            #           running_mode=FLAGS.running_mode),
            tf.contrib.training.SummaryAtEndHook(logdir),
            tf.contrib.training.StopAfterNEvalsHook(1)
        ],
        eval_ops=[printing_sequeces],
        max_number_of_evaluations=1,
        config=session_config)


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


if __name__ == '__main__':
    tf.app.run()
