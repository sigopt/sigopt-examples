"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from memn2n.data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np

from sigopt_memn2n_setup import sigopt_memn2n_experiment_setup
from sigopt_memn2n_setup.sigopt_memn2n_experiment_setup import ParametersList

import logging


def run_memn2n_single_training(tensorflow_commandline_flags):

    logging.info("Started Task: %s", str(tensorflow_commandline_flags.task_id))

    # preprocessing data before training memory network

    # task data
    train, test = load_task(tensorflow_commandline_flags.data_dir, tensorflow_commandline_flags.task_id)
    data = train + test

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    sentence_size = max(query_size, sentence_size)  # for the position
    sentence_size += 1  # +1 for time words

    sigopt_experiment_definition, e2e_memnn_experiment = sigopt_memn2n_experiment_setup.setup_sigopt_memn2n_experiment(tensorflow_commandline_flags)

    while e2e_memnn_experiment.progress.observation_count < e2e_memnn_experiment.observation_budget:
        logging.info("observation number: %d", e2e_memnn_experiment.progress.observation_count)

        suggestions = sigopt_experiment_definition.get_suggestions(e2e_memnn_experiment)

        memory_size = suggestions.assignments[ParametersList.MEMORY_SIZE.value]

        # Add time words/indexes
        for i in range(memory_size):
            word_idx['time{}'.format(i + 1)] = 'time{}'.format(i + 1)

        vocab_size = len(word_idx) + 1  # +1 for nil word

        logging.info("Longest sentence length %d", sentence_size)
        logging.info("Longest story length %d", max_story_size)
        logging.info("Average story length %d", mean_story_size)

        # train/validation/test sets
        S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
        trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1,
                                                                                     random_state=tensorflow_commandline_flags.random_state)
        testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

        # params
        n_train = trainS.shape[0]
        n_test = testS.shape[0]
        n_val = valS.shape[0]

        logging.info("Training Size %d", n_train)
        logging.info("Validation Size %d", n_val)
        logging.info("Testing Size %d", n_test)

        train_labels = np.argmax(trainA, axis=1)
        test_labels = np.argmax(testA, axis=1)
        val_labels = np.argmax(valA, axis=1)

        tf.set_random_seed(tensorflow_commandline_flags.random_state)
        batch_size = tensorflow_commandline_flags.batch_size

        batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
        batches = [(start, end) for start, end in batches]

        optimizer = sigopt_memn2n_experiment_setup.string_to_optimizer_object(suggestions.assignments[ParametersList.OPTIMIZER.value], suggestions.assignments)
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            model = MemN2N(batch_size, vocab_size, sentence_size,
                           memory_size=memory_size,
                           embedding_size=suggestions.assignments[ParametersList.WORD_EMBEDDING.value],
                           optimizer=optimizer,
                           session=sess,
                           hops=suggestions.assignments[ParametersList.HOP_SIZE.value],
                           max_grad_norm=tensorflow_commandline_flags.max_grad_norm)

            for t in range(1, tensorflow_commandline_flags.epochs+1):
                logging.info("epoch number: %d", t)
                logging.info("observation number: %d", e2e_memnn_experiment.progress.observation_count)

                np.random.shuffle(batches)
                total_cost = 0.0
                for start, end in batches:
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    a = trainA[start:end]
                    cost_t = model.batch_fit(s, q, a)
                    total_cost += cost_t

                if t % tensorflow_commandline_flags.evaluation_interval == 0:
                    train_preds = []
                    for start in range(0, n_train, batch_size):
                        end = start + batch_size
                        s = trainS[start:end]
                        q = trainQ[start:end]
                        pred = model.predict(s, q)
                        train_preds += list(pred)

                    val_preds = model.predict(valS, valQ)
                    train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
                    val_acc = metrics.accuracy_score(val_preds, val_labels)

                    test_preds = model.predict(testS, testQ)
                    test_acc = metrics.accuracy_score(test_preds, test_labels)

                    logging.info('-----------------------')
                    logging.info('Epoch %d', t)
                    logging.info('Total Cost: %f', total_cost)
                    logging.info('Training Accuracy: %f', train_acc)
                    logging.info('Validation Accuracy: %f', val_acc)
                    logging.info('Test Accuracy: %f', test_acc)
                    logging.info('-----------------------')

                    e2e_memnn_experiment, observation = sigopt_experiment_definition.update_experiment(e2e_memnn_experiment,
                                                                                          suggestions, test_acc)

        # reset computation graph to create new mm model
        tf.reset_default_graph()

    logging.info("Sig opt best parameters: %s", sigopt_experiment_definition.get_best_suggestions(e2e_memnn_experiment))


if __name__ == "__main__":
    tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
    tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
    tf.flags.DEFINE_integer("epochs", 3,
                            "Number of epochs to train for. Evaluation interval will be set to number of epochs.")
    tf.flags.DEFINE_integer("task_id", 17, "bAbI task id, 1 <= id <= 20")
    tf.flags.DEFINE_integer("random_state", None, "Random state.")
    tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")

    tf.flags.DEFINE_integer("sigopt_observation_budget", 3, "Define the observation budget for the SigOpt Experiment")
    tf.flags.DEFINE_string("sigopt_connection_token", None, "SigOpt API token")
    tf.flags.DEFINE_string("sigopt_experiment_name", "MemN2N", "Experiment name")

    FLAGS = tf.flags.FLAGS

    # set evaluation interval to number of epochs
    FLAGS.evaluation_interval = FLAGS.epochs

    run_memn2n_single_training(FLAGS)
