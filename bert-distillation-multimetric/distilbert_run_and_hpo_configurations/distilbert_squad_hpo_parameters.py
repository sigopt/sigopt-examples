from enum import Enum


class DistillationHyperparameter(Enum):
    ALPHA_CE = 'alpha_ce'
    ALPHA_SQUAD = 'alpha_squad'
    TEMPERATURE = 'temperature'


class SGDHyperparameter(Enum):
    LEARNING_RATE = 'learning_rate'
    WEIGHT_DECAY = 'weight_decay'
    ADAM_EPSILON = 'adam_epsilon'
    MAX_GRAD_NORM = 'max_grad_norm'
    WARM_UP_STEPS = 'warm_up_steps'
    BETA_1 = 'beta_1'
    BETA_2 = 'beta_2'
    PER_COMPUTE_TRAIN_BATCH_SIZE = 'per_compute_train_batch_size'
    PER_COMPUTE_EVAL_BATCH_SIZE = 'per_compute_eval_batch_size'
    GRADIENT_ACCUMULATION_STEPS = 'gradient_accumulation_steps'


class SquadArchitectureHyperparameter(Enum):
    MAX_SEQ_LENGTH = 'max_seq_length'  # for generating features from squad
    DOC_STRIDE = 'doc_stride'  # for generating features from squad
    MAX_QUERY_LENGTH = 'max_query_length'  # for generating features from squad
    MAX_ANSWER_LENGTH = 'max_answer_length'  # for calculating squad metrics


class ArchitectureHyperparameter(Enum):
    PRUNING_SEED = "pruning_seed" #seed set for pruning
    N_LAYERS = 'n_layers'  # DistilBert number of layers
    N_HEADS = 'n_heads'  # DistilBert number of attention heads
    DIMENSION = 'dim'
    HIDDEN_DIMENSION = 'hidden_dim'  # intermediate hidden layer dimension for ffn
    DROPOUT = "dropout"  # used for embeddings and ffn
    ATTENTION_DROPOUT = 'attention_dropout'  # dropout applied to attention heads
    INIT_RANGE = 'initializer_range'  # weight initialization for student model
    QA_DROPOUT = 'qa_dropout'  # dropout used for whole network for question answering


fine_tuning_squad2_default_hyperparameters = {
    DistillationHyperparameter.ALPHA_CE.value: 0.5,
    DistillationHyperparameter.ALPHA_SQUAD.value: 0.5,
    DistillationHyperparameter.TEMPERATURE.value: 1.0,
    SquadArchitectureHyperparameter.MAX_SEQ_LENGTH.value: 384,
    SquadArchitectureHyperparameter.MAX_QUERY_LENGTH.value: 64,
    SquadArchitectureHyperparameter.MAX_ANSWER_LENGTH.value: 30,
    SGDHyperparameter.LEARNING_RATE.value: float(5e-5),
    SGDHyperparameter.WEIGHT_DECAY.value: float(0.0),
    SGDHyperparameter.ADAM_EPSILON.value: float(1e-8),
    SGDHyperparameter.MAX_GRAD_NORM.value: float(1.0),
    SquadArchitectureHyperparameter.DOC_STRIDE.value: 128,
    SGDHyperparameter.PER_COMPUTE_TRAIN_BATCH_SIZE.value: 8,
    SGDHyperparameter.PER_COMPUTE_EVAL_BATCH_SIZE.value: 8,
    SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value: 1,
    SGDHyperparameter.WARM_UP_STEPS.value: 0,
    ArchitectureHyperparameter.N_HEADS.value: 12,
    ArchitectureHyperparameter.N_LAYERS.value: 6,
    ArchitectureHyperparameter.DIMENSION.value: 768,
    ArchitectureHyperparameter.HIDDEN_DIMENSION.value: 3072,
    ArchitectureHyperparameter.DROPOUT.value: 0.1,
    ArchitectureHyperparameter.ATTENTION_DROPOUT.value: 0.1,
    ArchitectureHyperparameter.INIT_RANGE.value: 0.02,
    ArchitectureHyperparameter.QA_DROPOUT.value: 0.1,
}


def get_default_hyperparameters():
    return fine_tuning_squad2_default_hyperparameters


def get_all_hyperparameter_names():
    all_hyperparameter_names = list()
    for distill_parameter in DistillationHyperparameter:
        all_hyperparameter_names.append(distill_parameter.value)
    for sgd_parameter in SGDHyperparameter:
        all_hyperparameter_names.append(sgd_parameter.value)
    for squad_arch_parameter in SquadArchitectureHyperparameter:
        all_hyperparameter_names.append(squad_arch_parameter.value)
    for arch_hyperparameter in ArchitectureHyperparameter:
        all_hyperparameter_names.append(arch_hyperparameter.value)
    return all_hyperparameter_names
