from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=100,
                    help='Number of images to process in a batch')

parser.add_argument('--data_dir', type=str, default='../dataset/mnist',
                    help='Path to the MNIST data directory.')

parser.add_argument('--model_dir', type=str, default='./model',
                    help='The directory where the model will be stored.')

parser.add_argument('--steps', type=int, default=20000,
                    help='Number of steps to train.')

def input_fn(mode, batch_size=1):
    def example_parser(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw' : tf.FixedLenFeature([], tf.string),
                'label' : tf.FixedLenFeature([], tf.int64)

            })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([28 * 28])

        image = tf.cast(image, tf.float32) / 255 - 0.5
        label = tf.cast(features['label'], tf.int32)
        return image, tf.one_hot(label, 10)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    else:
        assert mode == tf.estimator.ModeKeys.EVAL, 'invalid mode'
        tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')

    assert tf.gfile.Exists(tfrecords_file), (
        'Run convert_to_records.py first to convert the MNIST data to TFRecord '
        'file format.')

    dataset = tf.contrib.data.TFRecordDataset([tfrecords_file])

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat()

    dataset = dataset.map(example_parser,
                          num_threads=1,
                          output_buffer_size=batch_size)
    dataset = dataset.batch(batch_size)
    images, labels = dataset.make_one_shot_iterator().get_next()

    return images, labels

def mnist_model(inputs, mode):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])
    data_format = "channels_last"

    if tf.test.is_built_with_cuda():
        # NCHW better then NHWC
        data_format = "channels_first"
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    conv1 = tf.layers.conv2d(inputs = inputs,
                             filters = 32,
                             kernel_size = [5,5],
                             padding='same',
                             data_format=data_format,
                             activation=tf.nn.relu,
                             name = 'conv1')

    max_pool1 = tf.layers.max_pooling2d(inputs = conv1,
                                    pool_size = [2,2],
                                    strides = [2,2],
                                    padding='valid',
                                    data_format=data_format,
                                    name= 'max_pool1')

    conv2 = tf.layers.conv2d(inputs = max_pool1,
                             filters = 64,
                             kernel_size = [5,5],
                             padding = 'same',
                             data_format = data_format,
                             activation = tf.nn.relu,
                             name = 'conv2')

    max_pool2 = tf.layers.max_pooling2d(inputs = conv2,
                                        pool_size = [2, 2],
                                        strides = [2, 2],
                                        data_format = data_format,
                                        name = 'max_pool2')


    max_pool2_flat = tf.reshape(max_pool2,[-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs = max_pool2_flat,
                            units = 1024,
                            activity_regularizer = tf.nn.relu,
                            name = "dense")

    dropout = tf.layers.dropout(inputs = dense,
                                rate = 0.5,
                                training = (mode == tf.estimator.ModeKeys.TRAIN),
                                name= 'dropout')

    logits = tf.layers.dense(inputs = dropout,
                             units = 10,
                             activity_regularizer = tf.nn.relu,
                             name = "logits")
    return logits

def mnist_model_fn(features, labels, mode):
    logits = mnist_model(features, mode)

    predictions = {
        'classes' : tf.argmax(input=logits, axis=1),
        'probabilities' : tf.nn.softmax(logits=logits,name='softmax')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)
        train_op =  optimizer.minimize(loss=loss,
                                       global_step=tf.train.get_or_create_global_step())
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
                                   predictions=predictions['classes'])

    metrics = {'accuracy' : accuracy}

    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy',accuracy[1])


    return tf.estimator.EstimatorSpec(mode = mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics)

def main(unused_argv):
    mnist_classifier = tf.estimator.Estimator(model_fn=mnist_model_fn,
                                              model_dir=FLAGS.model_dir)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.SummaryWriter('./mnist_logs', sess.graph)
    summary_str = session.run(merged_summary_op)
    summary_writer.add_summary(summary_str, total_step)

    tensors_to_log = {
        'train_accuracy' : 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=100)

    mnist_classifier.train(input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN,FLAGS.batch_size),
                            hooks = [logging_hook],
                            steps = FLAGS.steps)

    eval_results = mnist_classifier.evaluate(input_fn=lambda: input_fn(tf.estimator.ModeKeys.EVAL))

    print('Evaluation results:\n    %s' % eval_results)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]] + unparsed)