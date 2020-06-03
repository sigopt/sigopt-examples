from squad_fine_tuning.train_squad_distillation import TrainSquadDistillation
from squad_fine_tuning.eval_squad_distillation import EvalSquadDistillation

import glob
import logging
import os

from logger import logger
from squad_fine_tuning.set_seed_and_dist import DEVICE
from distilbert_data_model_loaders.load_squad_dataset import load_and_cache_examples
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RunParameters

import torch

from transformers import (
    WEIGHTS_NAME,
)


class RunTrainingSquadDistillation(object):

    def __init__(self, model_class, tokenizer, tokenizer_class, teacher):
        self.model_class = model_class
        self.tokenizer = tokenizer
        self.tokenizer_class = tokenizer_class
        self.teacher = teacher

    def run_trainining_and_save(self, all_parameters, model, sigopt_run):
        logging.info("Training model")

        # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args_dict[
        # ArchitectureParameters.FP16.value] is set.
        # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
        # remove the need for this code, but it is still valid.
        if all_parameters[RunParameters.FP_16.value] is True:
            try:
                import apex

                apex.amp.register_half_function(torch, "einsum")
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        # Training
        if all_parameters[RunParameters.DO_TRAIN.value]:
            train_dataset = load_and_cache_examples(all_parameters, self.tokenizer, evaluate=False,
                                                    output_examples=False)
            train_squad_distillation = TrainSquadDistillation(model_class=self.model_class,
                                                    tokenizer=self.tokenizer,
                                   tokenizer_class=self.tokenizer_class, teacher=self.teacher)
            model, global_step, tr_loss = train_squad_distillation.train(model=model,
                                                                  args_dict=all_parameters,
                                                                  train_dataset=train_dataset,
                                                                  sigopt_run=sigopt_run)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save the trained model and the tokenizer
        if all_parameters[RunParameters.DO_TRAIN.value] and (all_parameters[RunParameters.LOCAL_RANK.value] == -1 or
                                                             torch.distributed.get_rank() == 0):
            self.save_trained_model(all_parameters, model)

            # Load a trained model and vocabulary that you have fine-tuned
            logging.info("loading trained model and vocabulary")
            model = self.model_class.from_pretrained(all_parameters[RunParameters.OUTPUT_DIR.value])
            tokenizer = self.tokenizer_class.from_pretrained(all_parameters[RunParameters.OUTPUT_DIR.value],
                                                        do_lower_case=all_parameters[
                                                            RunParameters.DO_LOWER_CASE.value])
            model.to(all_parameters[DEVICE])
            return model, tokenizer

    def save_trained_model(self, all_parameters, model):
        # Create output directory if needed
        if not os.path.exists(all_parameters[RunParameters.OUTPUT_DIR.value]) and all_parameters[RunParameters.LOCAL_RANK.value] in [-1, 0]:
            os.makedirs(all_parameters[RunParameters.OUTPUT_DIR.value])
        logger.info("Saving model checkpoints to %s", all_parameters[RunParameters.OUTPUT_DIR.value])
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(all_parameters[RunParameters.OUTPUT_DIR.value])
        self.tokenizer.save_pretrained(all_parameters[RunParameters.OUTPUT_DIR.value])
        # Good practice: save your training arguments together with the trained model
        torch.save(all_parameters,
                   os.path.join(all_parameters[RunParameters.OUTPUT_DIR.value], "training_args.bin"))

    def run_eval(self, all_parameters):
        # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
        results = {}
        eval_times = {}
        if all_parameters[RunParameters.DO_EVAL.value] and all_parameters[RunParameters.LOCAL_RANK.value] in [-1, 0]:
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
            checkpoints = self.load_checkpoints(all_parameters)

            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                logging.info("Evaluating checkpoint: {}".format(checkpoint))
                # Reload the model
                global_step = ""
                if "-" in checkpoint:
                    try:
                        global_step = int(checkpoint.split("-")[-1])
                    except ValueError:
                        logging.info("Checkpoint directory name not formatted as expected.")
                model = self.model_class.from_pretrained(checkpoint)
                model.to(all_parameters[DEVICE])

                # Evaluate
                eval_squad_distillation = EvalSquadDistillation(args_dict=all_parameters,
                                                                tokenizer=self.tokenizer,
                                                                global_step=global_step)
                dataset, examples, features = load_and_cache_examples(args_dict=all_parameters,
                                                                      tokenizer=self.tokenizer,
                                                                      evaluate=True,
                                                                      output_examples=True)
                result, eval_time = eval_squad_distillation.evaluate(model=model, dataset=dataset, examples=examples,
                                                                     features=features)

                result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
                results.update(result)

                if global_step == "":
                    eval_times.update({"last_checkpoint": eval_time})
                else:
                    eval_times.update({global_step: eval_time})
        logging.info("results from eval: {}".format(results))
        return results, eval_times

    def load_checkpoints(self, all_parameters):
        logger.info("Loading checkpoints")
        checkpoints = [all_parameters[RunParameters.OUTPUT_DIR.value]]
        if all_parameters[RunParameters.EVAL_ALL_CHECKPOINTS.value] is True:
            pytorch_bin_files = sorted(
                glob.glob(all_parameters[RunParameters.OUTPUT_DIR.value] + "/**/" + WEIGHTS_NAME,
                          recursive=True))
            for model_file in pytorch_bin_files:
                parent_directory = os.path.dirname(model_file)
                if parent_directory not in checkpoints:
                    checkpoints.append(parent_directory)
        return checkpoints
