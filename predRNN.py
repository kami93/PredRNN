from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.utils as utils
import tensorflow.keras.backend as backend

from keras_custom.layers.STLSTM import *

from pathlib import Path
import os
import time
########################################################################################################################
####################################### OPTION FOR TENSORFLOW ISSUE POST ###############################################
model_creation_device = '/cpu:0'
## '/cpu:0', '/cpu:1', '/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3' are possible values ######################################
########################################################################################################################
########################################################################################################################


## Training Data Spec ##
IMG_SIZE = 64
TOTAL_SAMPLES = 10000
NUM_TRAIN_SAMPLES = 8000
INPUT_LEN = 10
PRED_LEN = 9
TOTAL_LEN = INPUT_LEN + PRED_LEN

## Model Spec ##
NUM_CELL = 4
FILTERS = 128
KERNEL_SIZE = 3

## Training Setup ##
NUM_GPU = 4

## Data Feed Option ##
SHUFFLE_BUFFER_SIZE = NUM_TRAIN_SAMPLES
BATCH_SIZE = 8
EPOCHS = 10
BATCHES_PER_EPOCH = NUM_TRAIN_SAMPLES//BATCH_SIZE
BATCHES_PER_EPOCH_VALID = (TOTAL_SAMPLES - NUM_TRAIN_SAMPLES)//BATCH_SIZE

## Miscellaneous ##
ABS_PATH = os.path.abspath('./')

# Download the training data if not exist.
file = Path(ABS_PATH + '/dataset/moving_mnist.npy')
if not file.is_file():
  print('# Moving MNIST dataset do not exist. Beginning a download...')
  keras.utils.get_file(fname=ABS_PATH + '/dataset/moving_mnist.npy',
                       origin='http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy')

def input_fn():
  dataset = np.load(ABS_PATH + '/dataset/moving_mnist.npy')
  dataset = np.expand_dims(dataset.transpose([1,0,2,3]), axis=-1)
  x = np.concatenate((dataset[:NUM_TRAIN_SAMPLES,:INPUT_LEN,:,:,:], np.zeros_like(dataset[:NUM_TRAIN_SAMPLES,INPUT_LEN:TOTAL_LEN,:,:,:])), axis=1)
  y = dataset[:NUM_TRAIN_SAMPLES,1:,:,:,:]

  features = tf.data.Dataset.from_tensor_slices(x)
  features.prefetch(BATCH_SIZE)
  labels = tf.data.Dataset.from_tensor_slices(y)
  labels.prefetch(BATCH_SIZE)

  train_dataset = tf.data.Dataset.zip((features, labels))
  labels.prefetch(BATCH_SIZE)
  train_dataset = train_dataset.apply(tf.contrib.data.shuffle_and_repeat(SHUFFLE_BUFFER_SIZE, EPOCHS))
  labels.prefetch(BATCH_SIZE)
  train_dataset = train_dataset.map(lambda x,y: (backend.cast(x, 'float32'), backend.cast(y, 'float32')), num_parallel_calls=24)
  train_dataset = train_dataset.batch(BATCH_SIZE)
  train_dataset = train_dataset.prefetch(1)
  
  x_val = np.concatenate((dataset[NUM_TRAIN_SAMPLES:,:INPUT_LEN,:,:,:], np.zeros_like(dataset[NUM_TRAIN_SAMPLES:,INPUT_LEN:TOTAL_LEN,:,:,:])), axis=1)
  y_val = dataset[NUM_TRAIN_SAMPLES:,1:,:,:,:]

  features_val = tf.data.Dataset.from_tensor_slices(x_val)
  labels_val = tf.data.Dataset.from_tensor_slices(y_val)
  valid_dataset = tf.data.Dataset.zip((features_val, labels_val))
  valid_dataset = valid_dataset.repeat(EPOCHS)
  valid_dataset = valid_dataset.batch(BATCH_SIZE)
  valid_dataset = valid_dataset.prefetch(1)

  return train_dataset, valid_dataset

def l1_l2_loss(target, pred):
  diff = target - pred
  loss_ = tf.pow(diff, 2) + tf.abs(diff) # L2 + L1
  return backend.mean(loss_, axis=list(range(5)))

with keras.utils.custom_object_scope({'StackedSTLSTMCells':StackedSTLSTMCells,
                                      'STLSTMCell':STLSTMCell}): # Custom object scope for custom keras layers
  
  with tf.device(model_creation_device):
    ''' This is where the issue arises. '''
    # If model_creation_device == '/cpu:0',
    # Warning related to device placement is raised and
    # the training is extreamly slow with 100% CPU usage.

    # If model_creation_device == '/cpu:1', '/gpu:0', '/gpu:1', '/gpu:2', or '/gpu:3'
    # Warning related to device placement is raised,
    # Warning related to memory shortage is raised, and
    # GPU usage is unbalanced and highly concentrated to the model_creation device.
    # Except that the GPU usage is concentrated to gpu:0 when model_creation device == '/cpu:1'.
  
    cells = StackedSTLSTMCells([STLSTMCell(filters=FILTERS, kernel_size=KERNEL_SIZE) for _ in range(NUM_CELL)])
    predRNN = keras.Sequential([
      STLSTM2D(cells, return_sequences=True, input_shape=(TOTAL_LEN, IMG_SIZE, IMG_SIZE, 1)),
      keras.layers.Reshape(target_shape=(IMG_SIZE*TOTAL_LEN, IMG_SIZE, FILTERS)),
      keras.layers.Conv2D(filters=1, kernel_size=1),
      keras.layers.Reshape(target_shape=(TOTAL_LEN, IMG_SIZE, IMG_SIZE, 1))
      ])
    predRNN.summary()

  predRNN_multi = utils.multi_gpu_model(predRNN, gpus=NUM_GPU) # Make Multi GPU model.
  optimizer = keras.optimizers.Adam(lr=0.001)
  predRNN_multi.compile(optimizer = optimizer,
                        loss = l1_l2_loss,
                        metrics = [tf.keras.metrics.mse])
  
  train_dataset, valid_dataset = input_fn() # Make TF datasets
  
  callbacks = [] # Make training callback list
  callbacks.append(
    tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                   histogram_freq=0,
                                   batch_size=BATCH_SIZE,
                                   write_graph=True,
                                   write_grads=False))
  callbacks.append(
    tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/{epoch:02d}.hdf5',
                                       verbose=1,
                                       period=1))

  predRNN_multi.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    validation_data=valid_dataset,
                    steps_per_epoch=BATCHES_PER_EPOCH,
                    validation_steps=BATCHES_PER_EPOCH_VALID)

