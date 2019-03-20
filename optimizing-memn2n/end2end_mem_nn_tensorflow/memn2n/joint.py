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
from sigopt import Connection

from sigopt_memn2n_setup import sigopt_memn2n_experiment_setup
from sigopt_memn2n_setup.sigopt_memn2n_experiment_setup import ParametersList

import logging


def run_memn2n_joint_training(tensorflow_commandline_flags):

    # parsing commandline input for defining tasks to log with each observation
    calc_accuracy_tasks = []
    if tensorflow_commandline_flags.sigopt_calc_accuracy_tasks.split is not None:
        calc_accuracy_tasks = [int(i) for i in tensorflow_commandline_flags.sigopt_calc_accuracy_tasks.split(",")]

    # preprocessing data

    # load all train/test data
    ids = range(1, 21)
    train, test = [], []
    for i in ids:
        tr, te = load_task(tensorflow_commandline_flags.data_dir, i)
        train.append(tr)
        test.append(te)
    data = list(chain.from_iterable(train + test))

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    sentence_size = max(query_size, sentence_size)  # for the position
    sentence_size += 1  # +1 for time words

    logging.debug("setting up sigopt experiment")

    sigopt_experiment_definition, e2e_memnn_experiment = sigopt_memn2n_experiment_setup.setup_sigopt_memn2n_experiment(tensorflow_commandline_flags)

    while e2e_memnn_experiment.progress.observation_count < e2e_memnn_experiment.observation_budget:

        logging.info("starting new observation cycle")
        logging.info("observation number: %d", e2e_memnn_experiment.progress.observation_count)

        logging.debug("getting sigopt suggestions")
        suggestions = sigopt_experiment_definition.get_suggestions(e2e_memnn_experiment)

        memory_size = suggestions.assignments[ParametersList.MEMORY_SIZE.value]

        # Add time words/indexes
        for i in range(memory_size):
            word_idx['time{}'.format(i + 1)] = 'time{}'.format(i + 1)

        vocab_size = len(word_idx) + 1  # +1 for nil word

        logging.info("Longest sentence length %d", sentence_size)
        logging.info("Longest story length %d", max_story_size)
        logging.info("Average story length %d", mean_story_size)

        logging.info("transforming data")

        # train/validation/test sets
        trainS = []
        valS = []
        trainQ = []
        valQ = []
        trainA = []
        valA = []
        for task in train:
            S, Q, A = vectorize_data(task, word_idx, sentence_size, memory_size)
            ts, vs, tq, vq, ta, va = cross_validation.train_test_split(S, Q, A, test_size=0.1,
                                                                       random_state=tensorflow_commandline_flags.random_state)
            trainS.append(ts)
            trainQ.append(tq)
            trainA.append(ta)
            valS.append(vs)
            valQ.append(vq)
            valA.append(va)

        trainS = reduce(lambda a, b: np.vstack((a, b)), (x for x in trainS))
        trainQ = reduce(lambda a, b: np.vstack((a, b)), (x for x in trainQ))
        trainA = reduce(lambda a, b: np.vstack((a, b)), (x for x in trainA))
        valS = reduce(lambda a, b: np.vstack((a, b)), (x for x in valS))
        valQ = reduce(lambda a, b: np.vstack((a, b)), (x for x in valQ))
        valA = reduce(lambda a, b: np.vstack((a, b)), (x for x in valA))

        testS, testQ, testA = vectorize_data(list(chain.from_iterable(test)), word_idx, sentence_size, memory_size)

        n_train = trainS.shape[0]
        n_val = valS.shape[0]
        n_test = testS.shape[0]

        logging.info("Training Size: %d", n_train)
        logging.info("Validation Size: %d", n_val)
        logging.info("Testing Size: %d", n_test)

        train_labels = np.argmax(trainA, axis=1)
        test_labels = np.argmax(testA, axis=1)
        val_labels = np.argmax(valA, axis=1)

        tf.set_random_seed(tensorflow_commandline_flags.random_state)
        batch_size = tensorflow_commandline_flags.batch_size

        # This avoids feeding 1 task after another, instead each batch has a random sampling of tasks
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

            logging.info("batch training memory network")

            for i in range(1, tensorflow_commandline_flags.epochs+1):

                logging.debug("epoch number %d", i)
                logging.debug("observation count %d", e2e_memnn_experiment.progress.observation_count)

                np.random.shuffle(batches)

                total_cost = 0.0
                for start, end in batches:
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    a = trainA[start:end]
                    cost_t = model.batch_fit(s, q, a)
                    total_cost += cost_t

                if i % tensorflow_commandline_flags.evaluation_interval == 0:

                    logging.info("calculating training and validation accuracy.")

                    train_accs = []
                    for start in range(0, n_train, int(n_train/20)):
                        end = start + int(n_train/20)
                        s = trainS[start:end]
                        q = trainQ[start:end]
                        pred = model.predict(s, q)
                        acc = metrics.accuracy_score(pred, train_labels[start:end])
                        train_accs.append(acc)

                    logging.debug("Training accuracy %f", np.average(train_accs))

                    val_accs = []
                    for start in range(0, n_val, int(n_val/20)):
                        end = start + int(n_val/20)
                        s = valS[start:end]
                        q = valQ[start:end]
                        pred = model.predict(s, q)
                        acc = metrics.accuracy_score(pred, val_labels[start:end])
                        val_accs.append(acc)

                    logging.debug("Validation accuracy %f", np.average(val_accs))

                    test_accs = []
                    for start in range(0, n_test, int(n_test / 20)):
                        end = start + int(n_test / 20)
                        s = testS[start:end]
                        q = testQ[start:end]
                        pred = model.predict(s, q)
                        acc = metrics.accuracy_score(pred, test_labels[start:end])
                        test_accs.append(acc)

                    logging.info('-----------------------')
                    logging.info('Total Cost: %d', total_cost)

                    task_accuracies = []
                    t = 1
                    for t1, t2, t3 in zip(train_accs, val_accs, test_accs):
                        logging.info("Task %d", t)
                        logging.info("Training Accuracy %f", t1)
                        logging.info("Validation Accuracy %f", t2)
                        logging.info("Testing Accuracy %f", t3)
                        if t in calc_accuracy_tasks:
                            task_accuracies.append(t3)
                        t += 1
                    logging.info('-----------------------')

                    # log task test accuracies with current observation
                    metadata_dict = {}
                    for t, task_name in enumerate(calc_accuracy_tasks):
                        metadata_dict[str(task_name)] = task_accuracies[t]

                    test_accs_average = np.average(test_accs)

                    logging.debug("creating sigopt observation")
                    try:
                        e2e_memnn_experiment, current_observation = sigopt_experiment_definition.update_experiment_metadata(e2e_memnn_experiment,
                                                                                                               suggestions,
                                                                                                               test_accs_average, metadata_dict)
                    except ConnectionError as error:
                        logging.debug("connection problem: %s", str(error))
                        conn = Connection(client_token=tensorflow_commandline_flags.sigopt_connection_token)
                        conn.experiments(e2e_memnn_experiment.id).observations().create(suggestion=suggestions.id, value=test_accs_average, metadata=metadata_dict)
                        e2e_memnn_experiment = conn.experiments(e2e_memnn_experiment.id).fetch()

        tf.reset_default_graph()

    logging.info("Sig opt best parameters: %s", sigopt_experiment_definition.get_best_suggestions(e2e_memnn_experiment))


if __name__ == "__main__":
    tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
    tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
    tf.flags.DEFINE_integer("epochs", 1, "Number of epochs to train for.")
    tf.flags.DEFINE_integer("random_state", None, "Random state.")
    tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")

    tf.flags.DEFINE_integer("sigopt_observation_budget", 3, "Define the observation budget for the SigOpt Experiment")
    tf.flags.DEFINE_string("sigopt_connection_token", None, "SigOpt API token")
    tf.flags.DEFINE_string("sigopt_experiment_name", "MemN2N", "Experiment name")
    tf.flags.DEFINE_string("sigopt_calc_accuracy_tasks", "5,17,19", "Tasks used to measure optimizaiton progress. Default 5, 17, 19.")

    FLAGS = tf.flags.FLAGS

    # set evaluation interval to number of epochs
    FLAGS.evaluation_interval = FLAGS.epochs

    run_memn2n_joint_training(FLAGS)
