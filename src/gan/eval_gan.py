"""Evaluation of GAN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from gan.models import get_model
from gan.parameters import get_flags
from gan.protein.protein import BlastHook
from gan.documentation import setup_logdir, get_properties
from tensorflow.contrib.training import evaluate_repeatedly

slim = tf.contrib.slim
tfgan = tf.contrib.gan

FLAGS = get_flags()


def main(_):
    FLAGS.properties_file = "properties_test.json"
    properties = get_properties(FLAGS)
    logdir = setup_logdir(FLAGS, properties)
    FLAGS.running_mode = "test"
    model = get_model(FLAGS, properties)

    with tf.variable_scope('model', reuse=True):
        batch = model.data_handler.get_batch(FLAGS.batch_size, FLAGS)
        noise = tf.random_uniform([FLAGS.batch_size, FLAGS.z_dim], minval=-1.0, maxval=1.0, dtype=tf.float32, name='z0')
        generated_data = model.get_generated_data(noise, batch[1:])
        model.data_handler.display_fake_data(generated_data, batch[1], FLAGS.batch_size)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    id_to_enzyme_class_dict = properties["class_mapping"]
    evaluate_repeatedly(
        checkpoint_dir=logdir,
        hooks=[BlastHook(id_to_enzyme_class_dict, every_n_steps=1, output_dir=logdir, n_examples=FLAGS.batch_size,
                         running_mode=FLAGS.running_mode),
               tf.contrib.training.SummaryAtEndHook(logdir),
               tf.contrib.training.StopAfterNEvalsHook(1)
               ],
        eval_ops=generated_data,
        config=session_config)


if __name__ == '__main__':
    tf.app.run()
