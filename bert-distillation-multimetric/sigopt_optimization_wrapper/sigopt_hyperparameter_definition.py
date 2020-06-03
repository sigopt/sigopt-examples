import numpy as np
from sigopt_clients import sigopt_experiment_datatypes
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import ArchitectureHyperparameter,\
    DistillationHyperparameter, SGDHyperparameter

LOG_TRANSFORM_HYPERPARAMETERS = [SGDHyperparameter.LEARNING_RATE.value, SGDHyperparameter.ADAM_EPSILON.value,
                                 SGDHyperparameter.WEIGHT_DECAY.value]


def get_sigopt_hyperparameter_list():
    """Creates SigOpt formatted hyperparameter dictionary of enums and values above."""

    parameters_list = [
        dict(name=SGDHyperparameter.LEARNING_RATE.value, bounds=dict(min=np.log(2e-6), max=np.log(1e-1)),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=SGDHyperparameter.BETA_1.value, bounds=dict(min=0.7, max=0.9999),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=SGDHyperparameter.BETA_2.value, bounds=dict(min=0.7, max=0.9999),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=SGDHyperparameter.ADAM_EPSILON.value, bounds=dict(min=np.log(1e-10), max=np.log(1e-5)),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=SGDHyperparameter.WEIGHT_DECAY.value, bounds=dict(min=np.log(1e-7), max=np.log(1e-2)),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=DistillationHyperparameter.ALPHA_CE.value, bounds=dict(min=0.0000001, max=1.0),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=DistillationHyperparameter.TEMPERATURE.value, bounds=dict(min=1, max=10),
             type=sigopt_experiment_datatypes.SigOptDataTypes.INT.value),

        dict(name=ArchitectureHyperparameter.INIT_RANGE.value, bounds=dict(min=0.0, max=1.0),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=ArchitectureHyperparameter.ATTENTION_DROPOUT.value, bounds=dict(min=0.0, max=1.0),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=ArchitectureHyperparameter.QA_DROPOUT.value, bounds=dict(min=0.0, max=1.0),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=SGDHyperparameter.WARM_UP_STEPS.value, bounds=dict(min=0, max=100),
             type=sigopt_experiment_datatypes.SigOptDataTypes.INT.value),

        dict(name=ArchitectureHyperparameter.N_LAYERS.value, bounds=dict(min=1, max=20),
             type=sigopt_experiment_datatypes.SigOptDataTypes.INT.value),

        dict(name=ArchitectureHyperparameter.N_HEADS.value, bounds=dict(min=1, max=12),
             type=sigopt_experiment_datatypes.SigOptDataTypes.INT.value),

        dict(name=ArchitectureHyperparameter.PRUNING_SEED.value, bounds=dict(min=1, max=100),
             type=sigopt_experiment_datatypes.SigOptDataTypes.INT.value),

        dict(name=ArchitectureHyperparameter.DROPOUT.value, bounds=dict(min=0.0, max=1.0),
             type=sigopt_experiment_datatypes.SigOptDataTypes.DOUBLE.value),

        dict(name=SGDHyperparameter.PER_COMPUTE_TRAIN_BATCH_SIZE.value, bounds=dict(min=4, max=32),
             type=sigopt_experiment_datatypes.SigOptDataTypes.INT.value),

        dict(name=SGDHyperparameter.PER_COMPUTE_EVAL_BATCH_SIZE.value, bounds=dict(min=4, max=32),
             type=sigopt_experiment_datatypes.SigOptDataTypes.INT.value),
        ]

    return parameters_list
