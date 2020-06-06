import numpy as np
from sigopt_clients import sigopt_experiment_datatypes
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import ArchitectureHyperparameter,\
    DistillationHyperparameter, SGDHyperparameter

LOG_TRANSFORM_HYPERPARAMETERS = [SGDHyperparameter.LEARNING_RATE.value, SGDHyperparameter.ADAM_EPSILON.value,
                                 SGDHyperparameter.WEIGHT_DECAY.value]


def get_sigopt_hyperparameter_list():
    """Creates SigOpt formatted hyperparameter dictionary of enums and values above."""

    parameters_list = [
        dict(name=ArchitectureHyperparameter.ATTENTION_DROPOUT.value, bounds=dict(min=0.0, max=0.5),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=ArchitectureHyperparameter.DROPOUT.value, bounds=dict(min=0.0, max=0.5),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=SGDHyperparameter.LEARNING_RATE.value, bounds=dict(min=-14, max=-5),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=ArchitectureHyperparameter.N_HEADS.value, bounds=dict(min=1, max=12),
             type=sigopt_experiment_datatypes.SigOptDataTypes.INT.value),

        dict(name=ArchitectureHyperparameter.N_LAYERS.value, bounds=dict(min=1, max=10),
             type=sigopt_experiment_datatypes.SigOptDataTypes.INT.value),

        dict(name=ArchitectureHyperparameter.QA_DROPOUT.value, bounds=dict(min=0.0, max=0.5),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        ]

    return parameters_list
