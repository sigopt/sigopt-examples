from squad_fine_tuning.a_squad_w_distillation import ARunSquadDistillation
import logging
from logger import logger
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RunParameters
from squad_fine_tuning.set_seed_and_dist import setup_dist_training, set_seed, DEVICE
from squad_fine_tuning.training_run_squad_distillation import RunTrainingSquadDistillation
from squad_fine_tuning.a_squad_w_distillation import separate_config


class RunSquadDistillation(ARunSquadDistillation):

    def __init__(self, args_dict):
        super().__init__(args_dict)

    def run_training(self, run_training_squad_distillation, all_parameters, model):
        model, tokenizer = run_training_squad_distillation.run_trainining_and_save(all_parameters, model,
                                                                                   sigopt_run=None)
        return model, tokenizer

    def run_eval(self, run_training_squad_distillation, all_parameters):
        results, eval_times = run_training_squad_distillation.run_eval(all_parameters)
        return results, eval_times


def main(args_dict):

    logging.info("arguments passed to distillation training: {}".format(args_dict))

    args_dict, config_dict = separate_config(all_args_dict=args_dict)
    run_squad_distillation = RunSquadDistillation(args_dict=args_dict)
    run_squad_distillation.set_parameter_values(config_dict)
    run_squad_distillation.run_checks()

    # Setup CUDA, GPU & distributed training
    args_dict = setup_dist_training(args_dict)

    # Set seed
    set_seed(args_dict)

    # Load pretrained teacherm model, student model, and tokenizer
    teacher = run_squad_distillation.get_teacher()
    tokenizer = run_squad_distillation.get_tokenizer()
    student_loader, model, student_config = run_squad_distillation.get_student()

    model.to(args_dict[DEVICE])

    run_training_squad_distillation = RunTrainingSquadDistillation(model_class=student_loader.model_class,
                                                                   tokenizer=tokenizer,
                                                                   tokenizer_class=student_loader.tokenizer_class,
                                                                   teacher=teacher)
    all_parameters = dict()
    all_parameters.update(args_dict)
    all_parameters.update(config_dict)

    if args_dict[RunParameters.DO_TRAIN.value] is True:
        logging.info("running training and saving checkpoints")
        model, tokenizer = run_squad_distillation.run_training(run_training_squad_distillation, all_parameters,
                                                               model)

    logging.info("running evaluation")
    results, eval_time = run_squad_distillation.run_eval(run_training_squad_distillation, all_parameters)
    logger.info("Results: {}".format(results))

    return model, results, eval_time
