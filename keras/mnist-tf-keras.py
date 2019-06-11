'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Keras
Project: https://github.com/keras-team/keras

Modifications:
Modified to work with TF.Keras and to use IBM's Distributed Deep Learning library

*****************************************************************

Licensed Materials - Property of IBM

(C) Copyright IBM Corp. 2018. All Rights Reserved.

US Government Users Restricted Rights - Use, duplication or
disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

*****************************************************************
'''

from __future__ import print_function
from tensorflow.python import keras as keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K

import sys
import tensorflow as tf
if len(sys.argv) > 1 and sys.argv[1] == '--eager':
    tf.enable_eager_execution()
import ddl
import dataset


batch_size = 128
num_classes = 10

epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# data_dir
data_dir = "/tmp/mnist_convnet_model_data"+str(ddl.rank())

input_shape = ()
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


# the data, split between train and test sets
(train_set, num_of_train_imgs) = dataset.train(data_dir,input_shape)
train_set = train_set.shard(ddl.size(), ddl.rank())
train_set = train_set.cache().shuffle(buffer_size=1000).batch(batch_size).repeat()

(eval_set, num_of_test_imgs)  = dataset.test(data_dir, input_shape)
eval_full = eval_set
eval_set  = eval_set.shard(ddl.size(), ddl.rank())
eval_set  = eval_set.batch(batch_size).repeat()

num_of_all_test_imgs = num_of_test_imgs
num_of_train_imgs /= ddl.size()
num_of_test_imgs /= ddl.size()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# DDL: adjust learning rate based on number of GPUs.
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.train.AdadeltaOptimizer(learning_rate=1.0 * ddl.size()),
              metrics=['accuracy'])

callbacks = list()

# DDL: Add the DDL callback.
callbacks.append(ddl.DDLCallback())
callbacks.append(ddl.DDLGlobalVariablesCallback())

# DDL: Only use verbose = 1 on rank 0.
model.fit(train_set.make_one_shot_iterator(),
          steps_per_epoch=int(num_of_train_imgs // batch_size),
          epochs=epochs,
          verbose=1 if ddl.rank() == 0 else 0,
          validation_data=eval_set.make_one_shot_iterator(),
          validation_steps=int(num_of_test_imgs // batch_size),
          callbacks=callbacks)

# DDL: Only do final accuracy check on rank 0.
if ddl.rank() == 0:
  eval_full = eval_full.batch(batch_size).repeat()
  score     = model.evaluate(eval_full.make_one_shot_iterator(),
                             verbose=1,
                             steps=int(num_of_test_imgs))
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
