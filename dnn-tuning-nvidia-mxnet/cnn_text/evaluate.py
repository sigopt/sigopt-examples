import sys, time
import time

from collections import namedtuple

import mxnet as mx
import numpy as np

def evaluate_model(cnn_model, batch_size,
                   max_grad_norm, learning_rate,
                   epoch, x_train, y_train, x_dev,
                   y_dev):
    '''
    Train the cnn_model using back prop.
    '''
    optimizer='rmsprop'

    print 'optimizer', optimizer
    print 'maximum gradient', max_grad_norm
    print 'learning rate (step size)', learning_rate
    print 'epochs to train for', epoch

    # create optimizer
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate

    updater = mx.optimizer.get_updater(opt)

    # create logging output
    logs = sys.stderr

    # For each training epoch
    for iteration in range(epoch):
        tic = time.time()
        num_correct = 0
        num_total = 0

        # Over each batch of training data
        for begin in range(0, x_train.shape[0], batch_size):
            batchX = x_train[begin:begin+batch_size]
            batchY = y_train[begin:begin+batch_size]
            if batchX.shape[0] != batch_size:
                continue

            cnn_model.data[:] = batchX
            cnn_model.label[:] = batchY

            # forward
            cnn_model.cnn_exec.forward(is_train=True)

            # backward
            cnn_model.cnn_exec.backward()

            # eval on training data
            num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

            # update weights
            norm = 0
            for idx, weight, grad, name in cnn_model.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = np.sqrt(norm)
            for idx, weight, grad, name in cnn_model.param_blocks:
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)

                updater(idx, grad, weight)

                # reset gradient to zero
                grad[:] = 0.0

        # Decay learning rate for this epoch to ensure we are not "overshooting" optima
        if iteration % 50 == 0 and iteration > 0:
            opt.lr *= 0.5
            print >> logs, 'reset learning rate to %g' % opt.lr

        # End of training loop for this epoch
        toc = time.time()
        train_time = toc - tic
        train_acc = num_correct * 100 / float(num_total)

        # Evaluate model after this epoch on dev (test) set
        num_correct = 0
        num_total = 0

        # For each test batch
        for begin in range(0, x_dev.shape[0], batch_size):
            batchX = x_dev[begin:begin+batch_size]
            batchY = y_dev[begin:begin+batch_size]

            if batchX.shape[0] != batch_size:
                continue

            cnn_model.data[:] = batchX
            cnn_model.cnn_exec.forward(is_train=False)

            num_correct += sum(batchY == np.argmax(cnn_model.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        dev_acc = num_correct * 100 / float(num_total)
        print >> logs, 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f \
                --- Dev Accuracy thus far: %.3f' % (iteration, train_time, train_acc, dev_acc)
    return dev_acc
