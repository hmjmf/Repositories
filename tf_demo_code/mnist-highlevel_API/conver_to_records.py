from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

parser = argparse.ArgumentParser()

parser.add_argument('--directory', type=str, default='../dataset/mnist',
                    help='Directory to download data files and write the '
                    'converted result.')

parser.add_argument('--validation_size', type=int, default=100,
                    help='Number of examples to separate from the training '
                    'data for the validation set.')

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set, name):
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    assert num_examples == images.shape[0]

    rows = images.shape[1]
    cols = images.shape[2]
    channels = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')

    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        images_raw = images[index].tostring()
        example = tf.train.Example(features = tf.train.Features(feature={
            'height' : _int64_feature(rows),
            'width' : _int64_feature(cols),
            'channel' : _int64_feature(channels),
            'label' : _int64_feature(int(labels[index])),
            'image_raw' : _bytes_feature(images_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def main(unused_argv):
    data_sets = mnist.read_data_sets(FLAGS.directory,
                                     dtype=tf.uint8,
                                     reshape=False,
                                     validation_size=FLAGS.validation_size)
    convert_to(data_sets.train, 'train')
    convert_to(data_sets.validation, 'validation')
    convert_to(data_sets.test, 'test')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]] + unparsed)