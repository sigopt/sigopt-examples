from sigopt_memn2n_setup.sigopt_hyperparameters_enum import ParametersList
from sigopt_memn2n_setup.sigopt_hyperparameters_enum import SGDOptimizer
import numpy as np

parameters_list = [dict(name=ParametersList.LEARNING_RATE.value, bounds=dict(min=np.log(10e-7), max=np.log(1)), type="double"),
                   dict(name=ParametersList.MOMENTUM.value, bounds=dict(min=0.1, max=0.99), type="double"),
                   dict(name=ParametersList.NESTEROV.value, type="categorical", categorical_values=[dict(name="true"), dict(name="false")]),
                   dict(name=ParametersList.BETA_1.value, bounds=dict(min=0.8, max=0.99), type="double"),
                   dict(name=ParametersList.BETA_2.value, bounds=dict(min=0.95, max=0.9999), type="double"),
                   dict(name=ParametersList.EPSILON.value, bounds=dict(min=1e-9, max=1e-5), type="double"),
                   dict(name=ParametersList.DECAY_RATE.value, bounds=dict(min=0.9, max=0.99), type="double"),
                   dict(name=ParametersList.WORD_EMBEDDING.value, bounds=dict(min=10, max=100), type="int"),
                   dict(name=ParametersList.MEMORY_SIZE.value, bounds=dict(min=1, max=50), type="int"),
                   dict(name=ParametersList.HOP_SIZE.value, bounds=dict(min=1, max=3), type="int"),
                   dict(name=ParametersList.OPTIMIZER.value, type="categorical",
                        categorical_values=[dict(name=SGDOptimizer.ADAGRAD.value), dict(name=SGDOptimizer.GRADIENT_DESCENT_MOMENTUM.value),
                                            dict(name=SGDOptimizer.RMSPROP.value), dict(name=SGDOptimizer.ADAM.value),
                                            dict(name=SGDOptimizer.ADADELTA.value)])
                   ]
metrics_list = [dict(name="accuracy")]

conditionals_list = []

experiment_type = "random"
