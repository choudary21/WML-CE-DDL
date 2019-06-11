# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MNIST model training with TensorFlow eager execution.

See:
https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html

This program demonstrates training of the convolutional neural network model
defined in mnist.py with eager execution enabled.

If you are not interested in eager execution, you should ignore this file.

Modifications:
Modified to use IBM's Distributed Deep Learning library
Original: https://github.com/tensorflow/models/blob/master/official/mnist/mnist_eager.py

*****************************************************************

Licensed Materials - Property of IBM

(C) Copyright IBM Corp. 2018, 2019. All Rights Reserved.

US Government Users Restricted Rights - Use, duplication or
disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

*****************************************************************
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
tf.enable_eager_execution()
import ddl
import dataset as mnist_dataset
import shutil

tfe = tf.contrib.eager

batch_size = 100
train_epochs = 3

def create_model(data_format):
  """Model to recognize digits in the MNIST dataset.
  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
  But uses the tf.keras API.
  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(
              target_shape=input_shape,
              input_shape=(28 * 28,)),
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])


def loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def train(model, optimizer, dataset, step_counter, log_interval=None):
    """Trains model on `dataset` using `optimizer`."""

    start = time.time()
    for (batch, (images, labels)) in enumerate(dataset):
        with tf.contrib.summary.record_summaries_every_n_global_steps(
                10, global_step=step_counter):
            # Record the operations used to compute the loss given the input,
            # so that the gradient of the loss with respect to the variables
            # can be computed.
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss_value = loss(logits, labels)
                tf.contrib.summary.scalar('loss', loss_value)
                tf.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))
            grads = tape.gradient(loss_value, model.variables)
            optimizer.apply_gradients(
                zip(grads, model.variables), global_step=step_counter)
            if log_interval and batch % log_interval == 0 and ddl.rank() == 0:
                rate = log_interval / (time.time() - start)
                print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
                start = time.time()


def test(model, dataset):
    """Perform an evaluation of `model` on the examples from `dataset`."""
    avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
    accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

    for (images, labels) in dataset:
        logits = model(images, training=False)
        avg_loss(loss(logits, labels))
        accuracy(
            tf.argmax(logits, axis=1, output_type=tf.int64),
            tf.cast(labels, tf.int64))
    if ddl.rank() == 0:
        print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
              (avg_loss.result(), 100 * accuracy.result()))
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', avg_loss.result())
        tf.contrib.summary.scalar('accuracy', accuracy.result())


def run_mnist_eager():
    """Run MNIST training and eval loop in eager mode.
    """

    data_dir = '/tmp/tensorflow/mnist/input_data' + str(ddl.rank())
    model_dir = '/tmp/tensorflow/mnist/checkpoints/' + str(ddl.rank()) + '/'

    # Delete model dir
    if os.path.isdir(model_dir) and ddl.local_rank() == 0:
        shutil.rmtree(model_dir)

    data_format = 'channels_first'

    # Load the datasets
    train_ds, _ = mnist_dataset.train(data_dir, (1, 28, 28), label_int=True)
    train_ds = train_ds.shard(ddl.size(), ddl.rank()).shuffle(60000).batch(batch_size)
    test_ds, _ = mnist_dataset.test(data_dir, (1, 28, 28), label_int=True)
    test_ds = test_ds.batch(batch_size)

    # Create the model and optimizer
    model = create_model(data_format)
    optimizer = tf.train.MomentumOptimizer(0.01, 0.5)

    train_dir = None
    test_dir = None
    summary_writer = tf.contrib.summary.create_file_writer(
        train_dir, flush_millis=10000)
    test_summary_writer = tf.contrib.summary.create_file_writer(
        test_dir, flush_millis=10000, name='test')

    # Create and restore checkpoint (if one exists on the path)
    checkpoint_prefix = os.path.join(model_dir, 'ckpt-r' + str(ddl.rank()))
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, step_counter=step_counter)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(tf.train.latest_checkpoint(model_dir))

    # Train and evaluate for a set number of epochs.
    for _ in range(train_epochs):
        start = time.time()
        with summary_writer.as_default():
            train(model, optimizer, train_ds, step_counter,
                  10)
        end = time.time()
        if ddl.rank() == 0:
            print('\nTrain time for epoch #%d (%d total steps): %f' %
                  (checkpoint.save_counter.numpy() + 1,
                   step_counter.numpy(),
                   end - start))
        with test_summary_writer.as_default():
            test(model, test_ds)
        checkpoint.save(checkpoint_prefix)


if __name__ == '__main__':
    run_mnist_eager()
