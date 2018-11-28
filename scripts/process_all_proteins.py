#!/usr/bin/env python3
import argparse
import os
import time
from multiprocessing import Pool, freeze_support, cpu_count
import logging
import pandas as pd
from bio.amino_acid import filter_non_standard_amino_acids
from bio.amino_acid import from_amino_acid_to_id
import tensorflow as tf

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--source_file",
                    default="../data/protein/embedding/data_sources/unreviewed.gz",
                    type=str,
                    required=False,
                    help="Path the source file of data to process")
parser.add_argument("--destination",
                    default="../data/protein/embedding/sample/eval",
                    type=str,
                    required=False,
                    help="Path the folder the to store results")

parser.add_argument("--chunksize",
                    default=50000,
                    type=int,
                    help="The size of chunk")

parser.add_argument("--column_id",
                    default=1,
                    type=int,
                    help="Column id to use")

args = parser.parse_args()

logging.basicConfig(format="%(asctime)s - %(thread)-5d - %(levelname)-7s - %(message)s", level=logging.INFO)


def foo(data, path, chucksize):
    try:
        logging.debug("Starting thread")
        threading_start = time.time()
        data = filter_non_standard_amino_acids(data, "Sequence")
        data = from_amino_acid_to_id(data, "Sequence")
        filename = "{}/{}.{}".format(path, str(data.index[0] // chucksize), "tfrecords.gz")
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        with tf.python_io.TFRecordWriter(filename, options) as writer:
            for row in data:
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(row)])),
                        'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=row))
                    }
                ))
                writer.write(example.SerializeToString())
        logging.info("Data was stored in {} (Took: {}s)".format(filename, time.time() - threading_start))
    except Exception as e:
        logging.error("Something went wrong went writting in to tfrecords file")
        logging.error(e)


def get_source_data(args):
    return pd.read_csv(args.source_file, sep='\t', chunksize=args.chunksize, skipinitialspace=True,
                       usecols=[args.column_id])


if __name__ == '__main__':
    freeze_support()
    start_time = time.time()
    os.makedirs(args.destination, exist_ok=True)
    original_data = get_source_data(args)
    with Pool(processes=cpu_count(), maxtasksperchild=1) as pool:
        has_next = True
        while has_next:
            try:
                results = []
                for i in range(cpu_count()):
                    data = next(original_data)
                    results.append(pool.apply_async(foo, (data, args.destination, args.chunksize)))
            except Exception as e:
                logging.error(e)
                has_next = False
            finally:
                output = [p.get() for p in results]

    logging.info("Completed all threads in {} seconds".format(time.time() - start_time))
