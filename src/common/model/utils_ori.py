import math
import multiprocessing
import os
import sympy
import tensorflow as tf
from model.ops import pad_up_to

gfile = tf.gfile


def get_batches(fn, data_dir, batch_size, cycle_length=1, shuffle_buffer_size=100000, running_mode='train', args=None,
                balance=True):
    """

    Args:
      fn: function that tells how to process tfrecord
      data_dir: The directory to read data from.
      batch_size: The number of elements in a single minibatch.
      cycle_length: The number of input elements to process concurrently in the dataset loader. (Default: 1)
      shuffle_buffer_size: The number of records to load before shuffling. (Default: 100000)
      running_mode: string that is used to determine from where the data should be loaded (Default: train)
      args: list of arguments that will be passed into function that processes tfrecord (Default: None)

    Returns:
      A batch worth of data.

    """

    print("Loading files from {}".format(data_dir))
    filenames = tf.gfile.Glob(os.path.join(data_dir, running_mode, "*.tfrecords"))
    print("Found {} file(s)".format(len(filenames)))
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    filename_dataset = filename_dataset.shuffle(len(filenames))
    prefetch = max(int(batch_size / cycle_length), 1)
    # Repeat data in the file for unlimited number. This solves class imbalance problem.
    def get_tfrecord_dataset(fn):
        tfrecord_dataset = tf.data.TFRecordDataset(fn).prefetch(prefetch)
        if balance:
            tfrecord_dataset =  tfrecord_dataset.repeat()
        return tfrecord_dataset

    dataset = filename_dataset.interleave(
        lambda fn: get_tfrecord_dataset(fn), cycle_length=cycle_length)
    print("Loading process will use {} CPUs".format(multiprocessing.cpu_count()))
    dataset = dataset.map(lambda x: fn(x, args), num_parallel_calls=multiprocessing.cpu_count()).prefetch(batch_size)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()





def extract_seq_and_label(record, args):
    """Extracts and preprocesses the sequences and label from the record.

    Args:
      record: tfrecord from file
      args: list of arguments

    Returns:
      Sequence and label as tensors

    """
    context_features, seq_features = tf.parse_single_sequence_example(
        serialized=record,
        context_features={
            'label': tf.FixedLenFeature([], tf.int64),
            'length': tf.FixedLenFeature([], tf.int64)
        },
        sequence_features={
            'sequence': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=False)
        }
    )
    seq = tf.cast(seq_features['sequence'], tf.int32, name="seq")
    length = tf.cast(context_features['length'], tf.int32)
    labels = tf.cast(context_features['label'], tf.int32, name="labels")

    seq = pad_up_to(seq, args[0], dynamic_padding=args[1])
    return seq, labels


def extract_image_and_label(record, args):
    """Extracts and preprocesses the image and label from the record

    Args:
      record: tfrecord from file
      args: list of arguments

    Returns:
      Image and label as tensors

    """
    features = tf.parse_single_example(
        record,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.image.decode_png(features['image/encoded'], channels=3, dtype=tf.uint8)
    image.set_shape(args[0])

    image = tf.cast(image, tf.float32) * (2. / 255) - 1.

    label = tf.cast(features['image/class/label'], tf.int32)

    return image, label


def squarest_grid_size(num_images):
    """Calculates the size of the most square grid for num_images.
    Calculates the largest integer divisor of num_images less than or equal to
    sqrt(num_images) and returns that as the width. The height is
    num_images / width.

    Args:
      num_images: he total number of images.

    Returns:
      A tuple of (height, width) for the image grid.

    """
    divisors = sympy.divisors(num_images)
    square_root = math.sqrt(num_images)
    width = 1
    for d in divisors:
        if d > square_root:
            break
        width = d
    return (num_images // width, width)
