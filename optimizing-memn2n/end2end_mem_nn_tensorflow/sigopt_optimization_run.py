import tensorflow as tf
from memn2n import joint, single
import logging

# specify output file for logs
tf.flags.DEFINE_string("log_file", "memn2n_optimization_run.log",
                       "file path. Default will output file: memn2n_optimization_run.log in working directory")

# specify to run singly or jointly trained experiments
tf.flags.DEFINE_boolean("run_single_exp", None, "running a single experiment with task defined. False by default.")
tf.flags.DEFINE_boolean("run_joint_exp", None, "running joint experiment. True by default.")

# for joint training only
tf.flags.DEFINE_string("sigopt_calc_accuracy_tasks", "5,17,19",
                       "Tasks used to measure optimizaiton progress in increasing order. Default 5, 17, 19.")

# for single training only
tf.flags.DEFINE_integer("task_id", None, "bAbI task id, 1 <= id <= 20")

tf.flags.DEFINE_integer("evaluation_interval", None, "equal to number of epochs")

# common flags for memn2n experiments
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm. Default 40")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training. Default 32")
tf.flags.DEFINE_integer("epochs", 60, "Number of epochs to train for. Default 60.")
tf.flags.DEFINE_integer("random_state", None, "Random state seed. Default none")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/",
                       "Directory containing bAbI tasks. Default data/tasks_1-20_v1-2/en/")

# SigOpt experiment flags
tf.flags.DEFINE_integer("sigopt_observation_budget", None, "Define the observation budget for the SigOpt Experiment")
tf.flags.DEFINE_string("sigopt_connection_token", None, "SigOpt API token")
tf.flags.DEFINE_string("sigopt_experiment_name", None, "Experiment name")
tf.flags.DEFINE_string("sigopt_experiment_id", None, "Existing experiment id. If not none, will be used in experiment.")
tf.flags.DEFINE_string("experiment_type", None, "Must be: random, sigopt, or conditionals")

FLAGS = tf.flags.FLAGS

logging.basicConfig(filename=FLAGS.log_file, format='%(asctime)s %(message)s', level=logging.INFO)

assert (FLAGS.run_single_exp is True and FLAGS.run_joint_exp is None) or (FLAGS.run_single_exp is None and
                                                                          FLAGS.run_joint_exp is True)

# set evaluation interval to number of epochs
FLAGS.evaluation_interval = FLAGS.epochs

if FLAGS.run_joint_exp:
    joint.run_memn2n_joint_training(tensorflow_commandline_flags=FLAGS)

if FLAGS.run_single_exp:
    single.run_memn2n_single_training(tensorflow_commandline_flags=FLAGS)
