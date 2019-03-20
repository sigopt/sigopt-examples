import tensorflow as tf
import numpy as np
from sigopt_memn2n_setup.sigopt_experiment_client import SigOptExperiment
from sigopt_memn2n_setup.sigopt_hyperparameters_enum import ParametersList
from sigopt_memn2n_setup.sigopt_hyperparameters_enum import SGDOptimizer
from sigopt_memn2n_setup import sigopt_parameters_config, random_search_parameters_config, sigopt_conditionals_parameters_config
from enum import Enum
from sigopt import Connection


class ExperimentTypes(Enum):
    RANDOM = "random"
    SIGOPT = "sigopt"
    SIGOPT_CONDITIONALS = "conditionals"


class ConfigMapping(Enum):
    RANDOM = random_search_parameters_config
    SIGOPT = sigopt_parameters_config
    SIGOPT_CONDITIONALS = sigopt_conditionals_parameters_config


experiment_type_mapping = {ExperimentTypes.RANDOM: ConfigMapping.RANDOM,
 ExperimentTypes.SIGOPT_CONDITIONALS: ConfigMapping.SIGOPT_CONDITIONALS,
 ExperimentTypes.SIGOPT: ConfigMapping.SIGOPT}


def setup_sigopt_memn2n_experiment(tensorflow_commandline_flags):
    conn = Connection(client_token=tensorflow_commandline_flags.sigopt_connection_token)
    sigopt_experiment_definition = SigOptExperiment(conn)

    if tensorflow_commandline_flags.sigopt_experiment_id is None:

        # create metadata for experiment from commandline flags
        flags = tensorflow_commandline_flags.__dict__['__flags']
        metadata_dict = {}
        for name, commandline_value in flags.items():
            metadata_dict[name] = str(commandline_value)

        experiment_type = None
        for config_type in ExperimentTypes:
            if config_type.value == flags["experiment_type"]:
                experiment_type = config_type
                break

        config = experiment_type_mapping[experiment_type].value

        memn2n_experiment = sigopt_experiment_definition.initialize_experiment(
            experiment_name=tensorflow_commandline_flags.sigopt_experiment_name,
            parameters_list=config.parameters_list,
            metrics_list=config.metrics_list,
            conditionals_list=config.conditionals_list,
            observation_budget=tensorflow_commandline_flags.sigopt_observation_budget,
            metadata=metadata_dict,
            experiment_type=config.experiment_type)
    else:
        memn2n_experiment = sigopt_experiment_definition.get_initialized_experiment(tensorflow_commandline_flags.sigopt_experiment_id)

    return sigopt_experiment_definition, memn2n_experiment


def setup_adam_optimizer(sigopt_experiment_assignments):
    return tf.train.AdamOptimizer(learning_rate=np.exp(sigopt_experiment_assignments[ParametersList.LEARNING_RATE.value]),
                                  beta1=sigopt_experiment_assignments[ParametersList.BETA_1.value],
                                  beta2=sigopt_experiment_assignments[ParametersList.BETA_2.value],
                                  epsilon=sigopt_experiment_assignments[ParametersList.EPSILON.value])


def setup_adagrad(sigopt_experiment_assignments):
    return tf.train.AdagradOptimizer(learning_rate=np.exp(sigopt_experiment_assignments[ParametersList.LEARNING_RATE.value]))


def setup_gradient_descent_momentum(sigopt_experiment_assignments):
    return tf.train.MomentumOptimizer(learning_rate=np.exp(sigopt_experiment_assignments[ParametersList.LEARNING_RATE.value]),
                                                            momentum=sigopt_experiment_assignments[ParametersList.MOMENTUM.value],
                                                            use_nesterov=True if sigopt_experiment_assignments[ParametersList.NESTEROV.value] == "true" else False)


def setup_rmsprop(sigopt_experiment_assignments):
    return tf.train.RMSPropOptimizer(learning_rate=np.exp(sigopt_experiment_assignments[ParametersList.LEARNING_RATE.value]),
                                     decay=sigopt_experiment_assignments[ParametersList.DECAY_RATE.value],
                                     momentum=sigopt_experiment_assignments[ParametersList.MOMENTUM.value],
                                     epsilon=sigopt_experiment_assignments[ParametersList.EPSILON.value])


def setup_adadelta(sigopt_experiment_assignments):
    return tf.train.AdadeltaOptimizer(learning_rate=np.exp(sigopt_experiment_assignments[ParametersList.LEARNING_RATE.value]),
                                      epsilon=sigopt_experiment_assignments[ParametersList.EPSILON.value],
                                      rho=sigopt_experiment_assignments[ParametersList.DECAY_RATE.value])


def string_to_optimizer_object(string_argument, sigopt_experiment_assignments):
    if string_argument == SGDOptimizer.ADAM.value:
        return setup_adam_optimizer(sigopt_experiment_assignments)
    if string_argument == SGDOptimizer.ADAGRAD.value:
        return setup_adagrad(sigopt_experiment_assignments)
    if string_argument == SGDOptimizer.GRADIENT_DESCENT_MOMENTUM.value:
        return setup_gradient_descent_momentum(sigopt_experiment_assignments)
    if string_argument == SGDOptimizer.RMSPROP.value:
        return setup_rmsprop(sigopt_experiment_assignments)
    if string_argument == SGDOptimizer.ADADELTA.value:
        return setup_adadelta(sigopt_experiment_assignments)