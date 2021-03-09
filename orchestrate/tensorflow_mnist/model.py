import sigopt
import gzip
import math
import numpy as np
import os
from six.moves.urllib.request import urlretrieve
import tensorflow as tf

activation_functions = {
  'relu': tf.nn.relu,
  'sigmoid': tf.sigmoid,
  'tanh': tf.tanh,
}

optimizers = {
  'gradient_descent': tf.train.GradientDescentOptimizer,
  'rmsprop': tf.train.RMSPropOptimizer,
  'adam': tf.train.AdamOptimizer,
}

MNIST_HOST = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
IMAGE_WIDTH = 28
OUTPUT_CLASSES = 10

def load_onehot_data(filename):
  with gzip.open(filename, 'rb') as unzipped_file:
    data = np.frombuffer(unzipped_file.read(), dtype=np.uint8)
  labels = data[8:]
  length = len(labels)
  onehot = np.zeros((length, OUTPUT_CLASSES), dtype=np.float32)
  onehot[np.arange(length), labels] = 1
  return onehot

def load_image_data(filename):
  with gzip.open(filename, 'rb') as unzipped_file:
    data = np.frombuffer(unzipped_file.read(), dtype=np.uint8)
  images = data[16:].reshape((-1, IMAGE_WIDTH**2)).astype(np.float32)
  images /= 255
  return images

def load_mnist_data(path='/tmp/mnist'):
  if not os.path.isdir(path):
    os.makedirs(path)
  for data_file in [
    TRAIN_IMAGES,
    TRAIN_LABELS,
    TEST_IMAGES,
    TEST_LABELS,
  ]:
    destination = os.path.join(path, data_file)
    if not os.path.isfile(destination):
      urlretrieve("{}{}".format(MNIST_HOST, data_file), destination)
  return (
    (load_image_data(os.path.join(path, TRAIN_IMAGES)), load_onehot_data(os.path.join(path, TRAIN_LABELS))),
    (load_image_data(os.path.join(path, TEST_IMAGES)), load_onehot_data(os.path.join(path, TEST_LABELS))),
  )

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))

def conv_pool_block(x, filter_size, out_features, activation, pool_size):
  W = weight_variable([filter_size, filter_size, x.get_shape()[3].value, out_features])
  b = bias_variable([out_features])
  conv = activation_functions[activation](tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME') + b)
  pool = tf.nn.max_pool(conv, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')
  return pool

def fully_connected_layer(x, out_features):
  W = weight_variable([x.get_shape()[1].value, out_features])
  b = bias_variable([out_features])
  return tf.matmul(x, W) + b

def create_model():
  x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH**2])
  y = tf.placeholder(tf.float32, shape=[None, OUTPUT_CLASSES])
  keep_prob = tf.placeholder(tf.float32)
  x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_WIDTH, 1])

  block_1 = conv_pool_block(
    x_image,
    sigopt.get_parameter('conv_1_size', default=5),
    sigopt.get_parameter('conv_1_features', default=32),
    sigopt.get_parameter('conv_1_activation', default='relu'),
    sigopt.get_parameter('max_pool_1_size', default=2),
  )

  block_2 = conv_pool_block(
    block_1,
    sigopt.get_parameter('conv_2_size', default=5),
    sigopt.get_parameter('conv_2_features', default=64),
    sigopt.get_parameter('conv_2_activation', default='relu'),
    sigopt.get_parameter('max_pool_2_size', default=2),
  )

  _, block_2_height, block_2_width, block_2_features = block_2.get_shape()
  flattened = tf.reshape(block_2, [-1, block_2_height.value * block_2_width.value * block_2_features.value])

  fc_1 = activation_functions[sigopt.get_parameter('fc_activation', default='sigmoid')](
    fully_connected_layer(flattened, sigopt.get_parameter('fc_features', default=1024))
  )
  fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

  y_conv = fully_connected_layer(fc_1_drop, OUTPUT_CLASSES)
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
  optimizer = optimizers[sigopt.get_parameter('optimizer', default='adam')]
  learning_rate = 10**sigopt.get_parameter('log_learning_rate', default=-2)
  train_step = optimizer(learning_rate).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return x, y, y_conv, keep_prob, train_step, accuracy

def train_model(model, x_train, y_train):
  x, y, y_conv, keep_prob, train_step, _ = model
  train_length = len(x_train)
  batch_size = sigopt.get_parameter('batch_size', default=100)
  dropout_probability = sigopt.get_parameter('dropout_probability', default=0.2)
  for i in range(sigopt.get_parameter('epochs', default=1)):
    indices = np.arange(train_length)
    np.random.shuffle(indices)
    for start in range(0, train_length, batch_size):
      end = min(start + batch_size, train_length)
      batch_indices = indices[start:end]
      x_batch, y_batch = x_train[batch_indices], y_train[batch_indices]
      train_step.run(feed_dict={x: x_batch, y: y_batch, keep_prob: dropout_probability})

def evaluate_model(model, x_test, y_test):
  x, y, y_conv, keep_prob, _, accuracy = model
  return accuracy.eval(feed_dict={x: x_test, y: y_test, keep_prob: 1.0})

if __name__ == '__main__':
  from time import time
  (x_train, y_train), (x_test, y_test) = load_mnist_data()
  with tf.Session() as sess:
    model = create_model()
    sess.run(tf.global_variables_initializer())
    train_start = time()
    train_model(model, x_train, y_train)
    sigopt.log_metric('train_time', time() - train_start)
    eval_start = time()
    accuracy = evaluate_model(model, x_test, y_test)
    sigopt.log_metric('evaluation_time',  time() - eval_start)
    sigopt.log_metric('accuracy', accuracy)
