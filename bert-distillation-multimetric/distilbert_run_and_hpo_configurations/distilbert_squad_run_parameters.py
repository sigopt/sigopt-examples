from enum import Enum


class RunParameters(Enum):
    MODEL_TYPE = 'model_type'
    MODEL_NAME_OR_PATH = 'model_name_or_path'
    OUTPUT_DIR = 'output_dir'
    TEACHER_TYPE = 'teacher_type'
    TEACHER_NAME_OR_PATH = 'teacher_name_or_path'
    TRAIN_FILE = 'train_file'
    PREDICT_FILE = 'predict_file'
    CONFIG_NAME = 'config_name'
    TOKENIZER_NAME = 'tokenizer_name'
    CACHE_DIR = 'cache_dir'
    VERSION_2 = 'version_2_with_negative'
    EVALUATE_DURING_TRAINING = 'evaluate_during_training'
    DO_LOWER_CASE = 'do_lower_case'
    SEED = 'seed'
    LOGGING_STEPS = 'logging_steps'
    DO_TRAIN = 'do_train'
    DO_EVAL = 'do_eval'
    SAVE_STEPS = 'save_steps'
    EVAL_ALL_CHECKPOINTS = 'eval_all_checkpoints'
    NO_CUDA = 'no_cuda'
    OVERWRITE_OUTPUT_DIR = 'overwrite_output_dir'
    OVERWRTIE_CACHE = 'overwrite_cache'
    LOCAL_RANK = 'local_rank'
    NUM_TRAIN_EPOCHS = 'num_train_epochs'
    MAX_STEPS = 'max_steps'
    N_BEST_SIZE = 'n_best_size'
    VERBOSE_LOGGING = 'verbose_logging'
    FP16_OPT_LEVEL = 'fp16_opt_level'
    SERVER_IP = 'server_ip'
    SERVER_PORT = 'server_port'
    FP_16 = 'use_bfloat16'
    LOAD_PRETRAINED_MODEL = "load_student_pretrained"
    LOAD_SEMI_PRETRAINED_MODEL = "load_student_semi_pretrained"
    CACHE_S3_BUCKET = "cache_s3_bucket"
    TRAIN_CACHE_S3_DIRECTORY = "train_cache_s3_directory"
    EVAL_CACHE_S3_DIRECTORY = "eval_cache_s3_directory"


class OptimizationRunParameters(Enum):
    EXPERIMENT_NAME = "experiment_name"
    API_TOKEN = "api_token"
    USE_HPO_DEFAULT_RANGES = "use_hpo_default_ranges"
    PROJECT_NAME = "project_name"
    SIGOPT_EXPERIMENT_ID = "sigopt_experiment_id"
    SIGOPT_OBSERVATION_BUDGET = "sigopt_observation_budget"
    SIGOPT_RUN_DIRECTORY = "sigopt_run_directory"
    STORE_S3 = "store_s3"
    S3_BUCKET = "s3_bucket"


class RayTuneRunParameters(Enum):
    MAX_CONCURRENT = "max_concurrent"
    PARALLEL = "parallel"
    NUM_CPU = "num_cpu"
    NUM_GPU = "num_gpu"
    RAY_ADDRESS = "ray_address"
    CLEAN_RAYTUNE_OUTPUT = "clean_raytune_output"
    RAY_OUTPUT_DIRECTORY = "raytune_output_directory"


fine_tuning_squad2_default_run_parameters = {
    RunParameters.SEED.value: 42,
    RunParameters.LOGGING_STEPS.value: 1000,
    RunParameters.SAVE_STEPS.value: 1000,
    RunParameters.VERSION_2.value: True,
    RunParameters.EVALUATE_DURING_TRAINING.value: True,
    RunParameters.DO_LOWER_CASE.value: True,
    RunParameters.DO_EVAL.value: True,
    RunParameters.DO_TRAIN.value: True,
    RunParameters.EVAL_ALL_CHECKPOINTS.value: False,
    RunParameters.NO_CUDA.value: False,
    RunParameters.OVERWRITE_OUTPUT_DIR.value: False,
    RunParameters.OVERWRTIE_CACHE.value: False,
    RunParameters.LOCAL_RANK.value: -1,
    RunParameters.NUM_TRAIN_EPOCHS.value: 3,
    RunParameters.N_BEST_SIZE.value: 20,
    RunParameters.FP_16.value: False,
    RunParameters.FP16_OPT_LEVEL.value: '01',
    RunParameters.TRAIN_CACHE_S3_DIRECTORY.value: None,
    RunParameters.EVAL_CACHE_S3_DIRECTORY.value: None,
    RunParameters.CACHE_S3_BUCKET.value: None,
}


def get_default_run_parameters():
    return fine_tuning_squad2_default_run_parameters


def get_all_run_parameter_names():
    all_run_parameter_names = list()
    for run_parameter in RunParameters:
        all_run_parameter_names.append(run_parameter.value)
    for opt_run_parameter in OptimizationRunParameters:
        all_run_parameter_names.append(opt_run_parameter.value)
    return all_run_parameter_names
