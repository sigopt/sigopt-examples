import logging
import os
import timeit

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers.data.metrics.squad_metrics import compute_predictions_logits, \
    squad_evaluate
from transformers.data.processors.squad import SquadResult

from squad_fine_tuning.set_seed_and_dist import N_GPU, DEVICE
from logger import logger
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import SGDHyperparameter, SquadArchitectureHyperparameter
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RunParameters

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)


class EvalSquadDistillation(object):

    def __init__(self, args_dict, tokenizer, global_step):
        self.args_dict = args_dict
        self.tokenizer = tokenizer
        self.global_step = global_step
        self.output_directory = os.path.join(args_dict[RunParameters.OUTPUT_DIR.value],
                                             "checkpoint-{}".format(global_step))

        if not os.path.exists(self.output_directory) and self.args_dict[RunParameters.LOCAL_RANK.value] in [-1, 0]:
            os.makedirs(self.output_directory)

    def evaluate(self, model, dataset, examples, features):

        eval_batch_size, eval_dataloader = self.get_dataloader_sampler(dataset)

        # multi-gpu evaluate
        if self.args_dict[N_GPU] > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        else:
            model = model

        # Eval!
        logger.info("***** Running evaluation {} *****".format(self.global_step))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args_dict[eval_batch_size])

        all_results = []
        start_time = timeit.default_timer()
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.args_dict[DEVICE]) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                example_indices = batch[3]
                outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        eval_time = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset))

        # Compute predictions
        predictions = self.calcuate_predictions(all_results, examples, features)

        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
        return results, eval_time

    def calcuate_predictions(self, all_results, examples, features):
        output_prediction_file = os.path.join(self.output_directory,
                                              "predictions_{}.json".format(
                                                  self.global_step))
        output_nbest_file = os.path.join(self.output_directory,
                                         "nbest_predictions_{}.json".format(
                                             self.global_step))
        if self.args_dict[RunParameters.VERSION_2.value]:
            output_null_log_odds_file = os.path.join(self.output_directory,
                                                     "null_odds_{}.json".format(self.global_step))
        else:
            output_null_log_odds_file = None
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            self.args_dict[RunParameters.N_BEST_SIZE.value],
            self.args_dict[SquadArchitectureHyperparameter.MAX_ANSWER_LENGTH.value],
            self.args_dict[RunParameters.DO_LOWER_CASE.value],
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            self.args_dict[RunParameters.VERBOSE_LOGGING.value],
            self.args_dict[RunParameters.VERSION_2.value],
            self.args_dict["null_score_diff_threshold"],
            self.tokenizer,
        )
        return predictions

    def get_dataloader_sampler(self, dataset):
        eval_batch_size = "eval_batch_size"
        self.args_dict[eval_batch_size] = self.args_dict[SGDHyperparameter.PER_COMPUTE_EVAL_BATCH_SIZE.value] * max(1,
                                                                                                                    self.args_dict[
                                                                                                                        N_GPU])
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args_dict[eval_batch_size])
        return eval_batch_size, eval_dataloader


def to_list(tensor):
    return tensor.detach().cpu().tolist()
