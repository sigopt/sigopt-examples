"""
Loads and preprocessed data for the MR dataset.
Returns input vectors, labels, vocabulary, and inverse vocabulary.
"""
from collections import namedtuple

import numpy as np

from preprocess import (load_data_and_labels, pad_sentences,
                        build_vocab, build_input_data)

Data = namedtuple('Data', ['x_train', 'y_train', 'x_dev',
                           'y_dev', 'vocab_size', 'sentence_size'])

def get_data():
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    vocab_size = len(vocabulary)

    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split train/dev set
    # there are a total of 10662 labeled examples to train on
    x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

    sentence_size = x_train.shape[1]

    print 'Train/Dev split: %d/%d' % (len(y_train), len(y_dev))
    print 'train shape:', x_train.shape
    print 'dev shape:', x_dev.shape
    print 'vocab_size', vocab_size
    print 'sentence max words', sentence_size

    return Data(x_train, y_train, x_dev, y_dev, vocab_size, sentence_size)
