from squad_fine_tuning.optimize_squad_distillation import OptimizeSquadDistillation
import logging
import sigopt
from ray import tune
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import OptimizationRunParameters, RunParameters, RayTuneRunParameters
from squad_fine_tuning.a_squad_w_distillation import separate_config
import os
import boto3
import glob


class RayTuneOptimizeSquadDistillation(OptimizeSquadDistillation):

    def __init__(self, args_dict):
        super().__init__(args_dict)

    def sync_ray_checkpoints(self, args_dict):
        logging.info("Cleaning up checkpoint output directory")
        ray_output_directory = os.path.join(args_dict[RayTuneRunParameters.RAY_OUTPUT_DIRECTORY.value],
                                            args_dict[OptimizationRunParameters.EXPERIMENT_NAME.value])
        if args_dict[OptimizationRunParameters.STORE_S3.value] is True and os.path.exists(ray_output_directory):
            sync_ray_output(ray_output_directory, args_dict[OptimizationRunParameters.S3_BUCKET.value],
                            args_dict[RunParameters.OUTPUT_DIR.value],
                            args_dict[OptimizationRunParameters.SIGOPT_RUN_DIRECTORY.value])


def tune_track_metrics(evaluated_values):
    track_results_dict = dict()
    for a_metric_evaluation in evaluated_values:
        name = None
        value = None
        for dict_key, dict_value in a_metric_evaluation.items():
            if dict_key == "name":
                name = dict_value
            else:
                value = dict_value
        track_results_dict[name] = value
    logging.info("tune tracking eval dict: {}".format(track_results_dict))
    return track_results_dict


def sync_ray_output(ray_output_directory, s3_bucket, output_directory, sigopt_run_directory):
    logging.info("Looking for error files and experiment status files in: {}".format(ray_output_directory))
    s3_resource = boto3.resource('s3')
    error_files = glob.glob(os.path.join(ray_output_directory, "**", "error.txt"), recursive=True)
    logging.info("error files: {} found".format(error_files))
    if len(error_files) > 0:
        logging.info("Syncing error files to s3 bucket: {}".format(s3_bucket))
        for error_file in error_files:
            logging.info("Uploading {} to s3 Bucket {}".format(error_file, s3_bucket))
            s3_resource.Bucket(s3_bucket).upload_file(error_file, os.path.join(output_directory, "error.txt"))
    experiment_files = glob.glob(os.path.join(ray_output_directory, "experiment_state*.json"))
    if experiment_files is not None and len(experiment_files) > 0:
        experiment_file = glob.glob(os.path.join(ray_output_directory, "experiment_state*.json"))[0]
        s3_resource.Bucket(s3_bucket).upload_file(experiment_file, os.path.join(sigopt_run_directory,
                                                                                "ray_experiment_status.json"))


def main(ray_config_dict):
    args_dict, config_dict = separate_config(ray_config_dict)
    logging.info("arguments passed to distillation training: {}".format(args_dict))
    logging.info("configuration passed to distillation training: {}".format(config_dict))

    logging.info("setting SigOpt API token")
    os.environ["SIGOPT_API_TOKEN"] = args_dict[OptimizationRunParameters.API_TOKEN.value]
    os.environ["SIGOPT_API_URL"] = args_dict[OptimizationRunParameters.API_URL.value]

    tune_optimize_squad_distillation = RayTuneOptimizeSquadDistillation(args_dict=args_dict)
    suggestion_id = args_dict["suggestion_id"]
    parameter_values, args_dict, run_training_squad_distillation, model = tune_optimize_squad_distillation.setup_run(
        suggestion_id, args_dict, config_dict)

    all_parameters = dict()
    all_parameters.update(args_dict)
    all_parameters.update(parameter_values)

    failed, error_str, evaluated_values, results, model = tune_optimize_squad_distillation.try_distillation_tuning(
        run_training_squad_distillation,
        all_parameters,
        model,
        None)
    tune_track_dict = dict()
    tune_track_dict["error_str"] = error_str
    tune_track_dict["failed"] = failed
    if failed is True:
        pass
    else:
        track_results_dict = tune_track_metrics(evaluated_values)
        tune_track_dict.update(track_results_dict)
    tune.track.log(**tune_track_dict)
    tune_optimize_squad_distillation.sync_ray_checkpoints(args_dict)
    return model, evaluated_values, failed, error_str
