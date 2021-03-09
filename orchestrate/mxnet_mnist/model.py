import mxnet as mx
import numpy as np
import os
import struct
import logging

import sigopt

logging.getLogger().setLevel(logging.DEBUG)

mx_context = mx.gpu if os.environ.get('NVIDIA_VISIBLE_DEVICES') else mx.cpu

def unpack_X(filename):
    with open(filename) as f:
        magic, num, row, col = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8)
        X = X.reshape(num, 1, row, col)
        return X

def unpack_y(filename):
    with open(filename) as f:
        magic, num = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.int8)
        return y


X_train = unpack_X('data/train-images')
y_train = unpack_y('data/train-labels')
X_test = unpack_X('data/test-images')
y_test = unpack_y('data/test-labels')

batch_size = 100

def create_model():
    train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)

    data = mx.sym.Variable('data')

    conv1_kernel = sigopt.get_parameter('conv1_kernel', default=5)
    conv1_filters = sigopt.get_parameter('conv1_filters', default=10)
    conv1_act = sigopt.get_parameter('conv1_act', default='relu')
    conv1 = mx.sym.Convolution(
        data=data,
        kernel=(conv1_kernel, conv1_kernel),
        num_filter=conv1_filters,
    )
    act1 = mx.sym.Activation(data=conv1, act_type=conv1_act)
    pool1 = mx.sym.Pooling(data=act1, pool_type="max", kernel=(2, 2), stride=(2, 2))

    conv2_kernel = sigopt.get_parameter('conv2_kernel', default=5)
    conv2_filters = sigopt.get_parameter('conv2_filters', default=10)
    conv2_act = sigopt.get_parameter('conv2_act', default='relu')
    conv2 = mx.sym.Convolution(
        data=pool1,
        kernel=(conv2_kernel, conv2_kernel),
        num_filter=conv2_filters,
    )
    act2 = mx.sym.Activation(data=conv2, act_type=conv2_act)
    pool2 = mx.sym.Pooling(data=act2, pool_type="max", kernel=(2, 2), stride=(2, 2))

    fc1_hidden = sigopt.get_parameter('fc1_hidden', default=10)
    fc1_act = sigopt.get_parameter('fc1_act', default='relu')
    flatten = mx.sym.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=fc1_hidden)
    act3 = mx.sym.Activation(data=fc1, act_type=fc1_act)

    fc2 = mx.symbol.FullyConnected(data=act3, num_hidden=10)
    cnn = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

    model = mx.mod.Module(cnn, context=mx_context())

    model.fit(
        train_iter,
        eval_metric='acc',
        batch_end_callback=mx.callback.Speedometer(1, 100),
        # mxnet requires a string here, but orchestrate returns unicode in python2.7
        optimizer=str(sigopt.get_parameter('optimizer', default='adam')),
        optimizer_params={
          'learning_rate': 10**sigopt.get_parameter('log_learning_rate', default=-3)
        },
        num_epoch=sigopt.get_parameter('epochs', default=1),
    )
    return model

def evaluate_model():
    model = create_model()

    val_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)
    score = model.score(val_iter, mx.metric.Accuracy())
    return score[0][1]

if __name__ == '__main__':
    sigopt.log_metric('accuracy', evaluate_model())
