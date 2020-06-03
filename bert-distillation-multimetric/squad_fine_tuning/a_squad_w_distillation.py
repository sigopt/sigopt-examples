import logging
import os
import random
from distilbert_data_model_loaders.load_transformer_from_scratch import get_model_from_scratch
from distilbert_data_model_loaders.load_transformer_pretrained import get_pretrained_model, get_pretrained_tokenizer
from distilbert_data_model_loaders.load_transformer_semi_pretrained import get_semi_pretrained_model
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import ArchitectureHyperparameter, DistillationHyperparameter
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RunParameters
from squad_fine_tuning.set_seed_and_dist import DEVICE
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import get_all_hyperparameter_names
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import get_all_run_parameter_names
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import get_default_hyperparameters


class ARunSquadDistillation(object):

    def __init__(self, args_dict):
        self.args_dict = args_dict
        self.parameter_values = None

    def get_teacher(self):
        if self.args_dict[RunParameters.TEACHER_TYPE.value] is not None:
            logging.info("loading teacher model")
            teacher_loader, teacher, _ = get_pretrained_model(model_type=self.args_dict[RunParameters.TEACHER_TYPE.value],
                                                              model_name_or_path=self.args_dict[
                                                                  RunParameters.TEACHER_NAME_OR_PATH.value],
                                                              cache_dir=self.args_dict[RunParameters.CACHE_DIR.value],
                                                              )
            teacher.to(self.args_dict[DEVICE])
        else:
            teacher = None
        return teacher

    def get_student(self, pruning_seed=42):
        if self.args_dict[RunParameters.LOAD_PRETRAINED_MODEL.value] is True:
            logging.info("loading student model from pretrained checkpoints")
            student_loader, model, student_config = get_pretrained_model(
                model_type=self.args_dict[RunParameters.MODEL_TYPE.value].lower(),
                model_name_or_path=self.args_dict[
                    RunParameters.MODEL_NAME_OR_PATH.value],
                cache_dir=self.args_dict[
                    RunParameters.CACHE_DIR.value],
                )
        elif self.args_dict[RunParameters.LOAD_SEMI_PRETRAINED_MODEL.value] is True:
            logging.info("setting random seed {} for pruning".format(pruning_seed))
            random.seed(pruning_seed)
            logging.info("loading student model using pretrained weights as much as possible")
            default_n_heads, parameter_n_heads, parameter_n_layers, parameter_values = restructure_n_head_parameter(
                self.parameter_values)
            logging.info("loading student model with default number of n_heads")
            student_loader, model, student_config = get_semi_pretrained_model(model_type=self.args_dict[
                RunParameters.MODEL_TYPE.value].lower(),
                                      model_name_or_path=self.args_dict[RunParameters.MODEL_NAME_OR_PATH.value],
                                      cache_dir=self.args_dict[RunParameters.CACHE_DIR.value],
                                      config_dict=parameter_values,
                                      )
            if parameter_n_heads < default_n_heads:
                logging.info("pruning n_heads from default: {} to requested: {}".format(default_n_heads,
                                                                                        parameter_n_heads))
                model = prune_heads(model, default_n_heads, parameter_n_heads, parameter_n_layers)
            else:
                logging.info("no heads pruned")
        else:
            logging.info("loading student model from scratch and given configuration dictionary")
            student_loader, model, student_config = get_model_from_scratch(model_type=self.args_dict[
                RunParameters.MODEL_TYPE.value].lower(),
                                                                           config_dict=self.parameter_values)
        return student_loader, model, student_config

    def get_tokenizer(self):
        logging.info("loading tokenizer for student model")
        tokenizer = get_pretrained_tokenizer(model_type=self.args_dict[RunParameters.MODEL_TYPE.value].lower(),
                                             model_name_or_path=self.args_dict[RunParameters.MODEL_NAME_OR_PATH.value],
                                             cache_dir=self.args_dict[RunParameters.CACHE_DIR.value],
                                             )
        return tokenizer

    def set_parameter_values(self, parameter_values):
        self.parameter_values = parameter_values

    def run_checks(self):

        logging.info("running checks on input arguments")

        check_dict = dict()
        check_dict.update(self.args_dict)
        if self.parameter_values is not None:
            check_dict.update(self.parameter_values)

        # check to make sure args have booleans and not strings
        logging.info("checking correct formatting for boolean flags")
        for parameter, value in check_dict.items():
            if value == "True" or value == "False":
                raise Exception("String value for {} when boolean should exist".format(parameter))

        # Checking output directory args
        logging.info("checking output directory arguments")
        if (
            os.path.exists(check_dict[RunParameters.OUTPUT_DIR.value])
            and os.listdir(check_dict[RunParameters.OUTPUT_DIR.value])
            and check_dict[RunParameters.DO_TRAIN.value]
            and not check_dict[RunParameters.OVERWRITE_OUTPUT_DIR.value]
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    check_dict[RunParameters.OUTPUT_DIR.value]
                )
            )
        logging.info("Checking distillation weights and teacher model type")
        if check_dict[RunParameters.TEACHER_TYPE.value] is not None:
            assert check_dict[RunParameters.TEACHER_NAME_OR_PATH.value] is not None
            assert check_dict[DistillationHyperparameter.ALPHA_CE.value] > 0.0
            assert check_dict[DistillationHyperparameter.ALPHA_CE.value] + check_dict[
                DistillationHyperparameter.ALPHA_SQUAD.value] > 0.0
            assert check_dict[
                       RunParameters.TEACHER_TYPE.value] != "distilbert", "We constraint teachers not to be of type DistilBERT."

        logging.info("all checks passed")


def separate_config(all_args_dict):
    logging.info("separating config into args and hpo parameters")
    args_dict = dict()
    config_dict = dict()
    all_hpo_names = get_all_hyperparameter_names()
    all_run_parameters = get_all_run_parameter_names()
    for all_args_key, all_args_value in all_args_dict.items():
        if all_args_key in all_hpo_names:
            config_dict[all_args_key] = all_args_value
        elif all_args_key in all_run_parameters:
            args_dict[all_args_key] = all_args_value
        else:
            logging.info("{} not in hpo or run parameters, adding to args_dict".format(all_args_key))
            args_dict[all_args_key] = all_args_value
    logging.info("arguments separated into run arguments: {} and config arguments: {}".format(args_dict, config_dict))
    return args_dict, config_dict


def restructure_n_head_parameter(parameter_values):
    default_n_heads = get_default_hyperparameters()[ArchitectureHyperparameter.N_HEADS.value]
    parameter_values = dict(parameter_values)
    parameter_n_heads = parameter_values[ArchitectureHyperparameter.N_HEADS.value]
    parameter_n_layers = parameter_values[ArchitectureHyperparameter.N_LAYERS.value]
    parameter_values[ArchitectureHyperparameter.N_HEADS.value] = default_n_heads
    return default_n_heads, parameter_n_heads, parameter_n_layers, parameter_values


def prune_heads(model, default_n_heads, parameter_n_heads, parameter_n_layers):
    pruning_dict = dict()
    num_heads_to_prune = default_n_heads - parameter_n_heads
    for ith_layer in range(0, parameter_n_layers):
        pruning_dict[ith_layer] = random.sample(range(0, default_n_heads), num_heads_to_prune)
    logging.info("Pruning the following heads in layers: {}".format(pruning_dict))
    model.prune_heads(pruning_dict)
    return model
