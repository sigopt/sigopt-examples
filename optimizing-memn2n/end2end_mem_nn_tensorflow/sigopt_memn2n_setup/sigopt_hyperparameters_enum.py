from enum import Enum
import tensorflow as tf


class ParametersList(Enum):
    WORD_EMBEDDING = "word_embedding_size"
    MEMORY_SIZE = "memory_size"
    HOP_SIZE = "hop_size"
    DECAY_RATE = "decay_rate"
    EPSILON = "epsilon"
    BETA_1 = "beta_1"
    BETA_2 = "beta_2"
    NESTEROV = "nesterov"
    MOMENTUM = "momentum"
    LEARNING_RATE = "learning_rate"
    OPTIMIZER = "optimizer"


class SGDOptimizer(Enum):
    ADAM = "Adam"
    ADAGRAD = "Adagrad"
    GRADIENT_DESCENT_MOMENTUM = "GradientDescentMomentum"
    RMSPROP = "RMSProp"
    ADADELTA = "Adadelta"


## mapping from string to tensorflow object
optimizer_mapping = {SGDOptimizer.ADAM.value: tf.train.AdamOptimizer, SGDOptimizer.ADAGRAD.value: tf.train.AdagradOptimizer,
                    SGDOptimizer.GRADIENT_DESCENT_MOMENTUM.value: tf.train.MomentumOptimizer,
                    SGDOptimizer.RMSPROP.value: tf.train.RMSPropOptimizer}