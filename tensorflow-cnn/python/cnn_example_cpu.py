import argparse
import datetime
import numpy
import time
import scipy
import sigopt
import scipy.io
import tensorflow as tf
import math
from skimage.color import rgb2gray
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

# load SVHN dataset
extra_X = scipy.io.loadmat("extra_32x32.mat")['X'].astype('float64')
extra_y = scipy.io.loadmat("extra_32x32.mat")['y'].astype('float64')
test_X = scipy.io.loadmat("test_32x32.mat")['X'].astype('float64')
test_y = scipy.io.loadmat("test_32x32.mat")['y'].astype('float64')
train_X = scipy.io.loadmat("train_32x32.mat")['X'].astype('float64')
train_y = scipy.io.loadmat("train_32x32.mat")['y'].astype('float64')

def convert_rgb2gray(X):
  X_gray = numpy.zeros((32, 32, X.shape[3]))
  for i in xrange(0, X.shape[3]):
    img_gray = rgb2gray(X[:, :, :, i])
    X_gray[:, :, i] = img_gray
  return X_gray

# convert all image data to grayscale
extra_X = convert_rgb2gray(extra_X)
test_X = convert_rgb2gray(test_X)
train_X = convert_rgb2gray(train_X)

image_w = 32
train_XZ = numpy.reshape(train_X.T, (train_X.shape[2], image_w * image_w))
test_XZ = numpy.reshape(test_X.T, (test_X.shape[2], image_w * image_w))
extra_XZ = numpy.reshape(extra_X.T, (extra_X.shape[2], image_w * image_w))

# normalize image pixel features
extra_XZ = preprocessing.scale(extra_XZ, axis=1)
train_XZ = preprocessing.scale(train_XZ, axis=1)
test_XZ = preprocessing.scale(test_XZ, axis=1)

# convert SVHN labels to one-hot format
one_hot_enc = preprocessing.OneHotEncoder(sparse=False)
test_yZ = one_hot_enc.fit_transform(test_y)
train_yZ = one_hot_enc.fit_transform(train_y)
extra_yZ = one_hot_enc.fit_transform(extra_y)

# stack train and extra on top of each other
extra_XZ = numpy.concatenate((extra_XZ, train_XZ), axis=0)
extra_yZ = numpy.concatenate((extra_yZ, train_yZ), axis=0)

# only consider 75% of this dataset for now
_, extra_XZ, _, extra_yZ = train_test_split(extra_XZ, extra_yZ, test_size=0.75, random_state=42)

# create SigOpt experiment
conn = sigopt.Connection()
experiment = conn.experiments().create(
  name='SVHN ConvNet',
  project='sigopt-examples',
  metrics=[dict(name='value', objective='maximize')],
  parameters=[
    {'name': 'filter1_w',      'type': 'int',    'bounds': {'min': 3,  'max': 10}},
    {'name': 'filter1_depth',  'type': 'int',    'bounds': {'min': 10, 'max': 64}},
    {'name': 'filter2_w',      'type': 'int',    'bounds': {'min': 3,  'max': 10}},
    {'name': 'filter2_depth',  'type': 'int',    'bounds': {'min': 10, 'max': 64}},
    {'name': 'drp_out_keep_p', 'type': 'double', 'bounds': {'min': 0.2, 'max': 1.0}},
    {'name': 'log_rms_lr',     'type': 'double', 'bounds': {'min': math.log(0.00001),
                                                            'max': math.log(1.0)}},
    {'name': 'rms_mom',        'type': 'double', 'bounds': {'min': 0.5, 'max': 1.0}},
    {'name': 'rms_decay',      'type': 'double', 'bounds': {'min': 0.5, 'max': 1.0}},
  ],
  observation_budget=100,
)

# SigOpt optimization loop
for jk in xrange(experiment.observation_budget):
  # SigOpt params
  suggestion = conn.experiments(experiment.id).suggestions().create()
  params = suggestion.assignments

  sess = tf.InteractiveSession()
  x = tf.placeholder(tf.float32, shape=[None, image_w * image_w])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])
  filter1_w = int(params['filter1_w'])
  filter1_depth = int(params['filter1_depth'])
  filter2_w = int(params['filter2_w'])
  filter2_depth = int(params['filter2_depth'])
  rms_lr = math.exp(params['log_rms_lr'])
  rms_mom = params['rms_mom']
  rms_decay = params['rms_decay']
  drp_out_keep_p = params['drp_out_keep_p']

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  W_conv1 = weight_variable([filter1_w, filter1_w, 1, filter1_depth])
  b_conv1 = bias_variable([filter1_depth])

  x_image = tf.reshape(x, [-1,image_w,image_w,1], name='reshape1')

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([filter2_w, filter2_w, filter1_depth, filter2_depth])
  b_conv2 = bias_variable([filter2_depth])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([8 * 8 * filter2_depth, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * filter2_depth], name='rehsape2')
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  train_step = tf.train.RMSPropOptimizer(rms_lr, decay=rms_decay, momentum=rms_mom).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # generate single CV fold to do hyperparam optimization
  train_XZ_2, valid_XZ, train_yZ_2, valid_yZ = train_test_split(extra_XZ, extra_yZ,
                                                                test_size=0.15, random_state=42)

  sess.run(tf.initialize_all_variables())
  # run SGD
  t0 = time.time()
  batch_size = 10000
  epoch_size = 1
  for k in xrange(epoch_size):
    for i in xrange(0,train_XZ_2.shape[0],batch_size):
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:train_XZ_2[i:(i + batch_size)], y_: train_yZ_2[i:(i + batch_size)], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x:train_XZ_2[i:(i + batch_size)], y_: train_yZ_2[i:(i + batch_size)], keep_prob: drp_out_keep_p})
  # chunk opt metric, so we don't OOM error
  opt_metric = 0.0
  opt_chunk = 10
  for i in xrange(0,valid_XZ.shape[0],valid_XZ.shape[0]/opt_chunk):
    chunk_acc = accuracy.eval(feed_dict={x: valid_XZ[i:(i + valid_XZ.shape[0] / opt_chunk)], y_: valid_yZ[i:(i + valid_XZ.shape[0] / opt_chunk)], keep_prob: 1.0})
    chunk_range = min(i + valid_XZ.shape[0]/opt_chunk, valid_XZ.shape[0]) - i
    chunk_perc = chunk_range / float(valid_XZ.shape[0])
    opt_metric += chunk_acc * chunk_perc
  print(opt_metric)
  print("Total Time :", (time.time() - t0))
  sess.close()

  # report to SigOpt
  conn.experiments(experiment.id).observations().create(
    suggestion=suggestion.id,
    value=float(opt_metric),
    value_stddev=0.05
  )
