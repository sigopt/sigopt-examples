from squad_fine_tuning.a_squad_w_distillation import ARunSquadDistillation
import logging
import os
import numpy as np
import shutil
import boto3
from sigopt_optimization_wrapper.sigopt_hyperparameter_definition import LOG_TRANSFORM_HYPERPARAMETERS
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import ArchitectureHyperparameter, DistillationHyperparameter
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RunParameters, OptimizationRunParameters
from sigopt_optimization_wrapper.sigopt_multimetric_definition import ResultsAttributes
from squad_fine_tuning.set_seed_and_dist import setup_dist_training, set_seed, DEVICE
from squad_fine_tuning.training_run_squad_distillation import RunTrainingSquadDistillation


class OptimizeSquadDistillation(ARunSquadDistillation):

    def __init__(self, args_dict):
        super().__init__(args_dict)

    def transform_parameters(self, parameter_values):

        def get_alpha_squad_value(alpha_ce_value):
            return 1 - float(alpha_ce_value)

        # apply transformations for suggested values
        logging.info("log transforming suggested values for: {}".format(LOG_TRANSFORM_HYPERPARAMETERS))
        for parameter_name, curr_parameter_value in parameter_values.items():
            if parameter_name in LOG_TRANSFORM_HYPERPARAMETERS:
                parameter_values[parameter_name] = np.exp(curr_parameter_value)

        # get weight value for alpha squad
        logging.info("calcualting alpha squad weight value")
        parameter_values[DistillationHyperparameter.ALPHA_SQUAD.value] = get_alpha_squad_value(parameter_values[
                                                                                                   DistillationHyperparameter.ALPHA_CE.value])
        logging.info("parameter values now transformed to: {}".format(parameter_values))
        return parameter_values

    def setup_parameters(self, config_dict):
        parameter_values = self.transform_parameters(config_dict)
        self.set_parameter_values(parameter_values)
        return parameter_values

    def set_run_output_directory(self, suggestion_id):
        run_output_directory = os.path.join(self.args_dict[OptimizationRunParameters.SIGOPT_RUN_DIRECTORY.value],
                                            str(suggestion_id))
        logging.info("output directory for current suggestion: {}".format(run_output_directory))
        self.args_dict[RunParameters.OUTPUT_DIR.value] = run_output_directory

    def clean_up_checkpoints(self, failed):
        # upload to s3 and delete local directory
        s3_bucket_name = self.args_dict[OptimizationRunParameters.S3_BUCKET.value]
        upload_s3 = self.args_dict[OptimizationRunParameters.STORE_S3.value]
        if failed is False:
            if upload_s3 is True and s3_bucket_name is not None:
                upload_run_output_to_s3(s3_bucket_name, self.args_dict[RunParameters.OUTPUT_DIR.value],
                                        parent_output_dir=self.args_dict[
                                            OptimizationRunParameters.SIGOPT_RUN_DIRECTORY.value])
        else:
            # if run failed, check if directory exists and delete
            if os.path.exists(self.args_dict[RunParameters.OUTPUT_DIR.value]) is True:
                shutil.rmtree(self.args_dict[RunParameters.OUTPUT_DIR.value])

    def try_distillation_tuning(self, run_training_squad_distillation, all_parameters, model, run):
        failed = False
        error_str = "No exception to report"
        evaluated_values = dict()
        results = dict()
        try:
            model, results, evaluated_values = self.run_distillation_tuning(run_training_squad_distillation,
                                                                        all_parameters, model, run)
            logging.info("Results: {}".format(evaluated_values))
        except AssertionError as assertion_error:
            failed, error_str = self.check_assertion_error(assertion_error)
        except RuntimeError as runtime_error:
            failed, error_str = self.check_cuda_mem_error(runtime_error)
        finally:
            logging.info("Cleaning up checkpoint output directory")
            self.clean_up_checkpoints(failed)
        return failed, error_str, evaluated_values, results, model

    def run_distillation_tuning(self, run_training_squad_distillation, all_parameters, model, sigopt_run):
        model, tokenizer = run_training_squad_distillation.run_trainining_and_save(all_parameters, model,
                                                                                   sigopt_run=sigopt_run)
        results, eval_times = run_training_squad_distillation.run_eval(all_parameters)
        evaluated_values = self.get_sigopt_formatted_metrics(results, eval_times, model)
        return model, results, evaluated_values

    def check_assertion_error(self, assertion_error):
        logging.info("An assertion error has occured: {}".format(str(assertion_error)))
        if is_matrix_comp_exception(assertion_error):
            logging.info("Assertion error is related to parameter definitions for dimension and n_heads. Will "
                         "report as failed back to SigOpt.")
            failed = True
            error_str = str(assertion_error)
            return failed, error_str
        else:
            raise assertion_error

    def check_cuda_mem_error(self, runtime_error):
        logging.info("A run time error has occured: {}".format(str(runtime_error)))
        if is_cuda_mem_exception(runtime_error):
            logging.info("Run time exception is related to Cuda memory issues. Will report as failed back to "
                         "SigOpt.")
            failed = True
            error_str = str(runtime_error)
            return failed, error_str
        else:
            raise runtime_error

    def get_sigopt_formatted_metrics(self, results, eval_times, model):
        final_eval_time = eval_times["last_checkpoint"]
        # calculate number of parameters for model
        total_num_model_parameters = calculate_num_model_parameters(model)
        # grab metrics for SigOpt
        evaluated_values = [dict(name=ResultsAttributes.EXACT.value,
                             value=results[ResultsAttributes.EXACT.value]),
                            dict(name=ResultsAttributes.NUM_PARAMETERS.value,
                                value=total_num_model_parameters),
                            dict(name=ResultsAttributes.F1.value,
                             value=results[ResultsAttributes.F1.value]),
                            dict(name=ResultsAttributes.INFERENCE_TIME.value,
                             value=final_eval_time)]
        return evaluated_values

    def setup_run(self, suggestion_id, args_dict, config_dict):
        self.set_run_output_directory(suggestion_id)

        parameter_values = self.setup_parameters(config_dict)
        self.run_checks()

        # Setup CUDA, GPU & distributed training
        args_dict = setup_dist_training(args_dict)

        # Set seed
        set_seed(args_dict)

        # Load pretrained teacherm model, student model, and tokenizer
        teacher = self.get_teacher()
        tokenizer = self.get_tokenizer()
        student_loader, model, student_config = self.get_student(pruning_seed=config_dict[
            ArchitectureHyperparameter.PRUNING_SEED.value])

        # reset seed
        set_seed(args_dict)

        logging.debug("model architecture: {}".format(model))
        model.to(args_dict[DEVICE])

        logging.info("running training, saving checkpoints, and running evaluation")
        run_training_squad_distillation = RunTrainingSquadDistillation(model_class=student_loader.model_class,
                                                                       tokenizer=tokenizer,
                                                                       tokenizer_class=student_loader.tokenizer_class,
                                                                       teacher=teacher)
        return parameter_values, args_dict, run_training_squad_distillation, model


def is_cuda_mem_exception(runtime_error):
    if "CUDA out of memory" in str(runtime_error):
        return True
    return False


def is_matrix_comp_exception(assertion_error):
    if "Model dimension is not comptatible with number of heads" in str(assertion_error):
        return True
    return False


def calculate_num_model_parameters(model):
    logging.info("calcuating number of trainable parameters in the model")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def upload_run_output_to_s3(s3_bucket_name, local_output_directory, parent_output_dir):
    s3_resource = boto3.resource('s3')
    for top_dirs, sub_dirs, files in os.walk(local_output_directory):
        for file in files:
            abs_file_path = os.path.join(top_dirs, file)
            logging.info("Uploading {} to s3 Bucket {}".format(abs_file_path, s3_bucket_name))
            s3_resource.Bucket(s3_bucket_name).upload_file(abs_file_path, abs_file_path)

    logging.info("uploading files for final eval run from parent directory")
    for file in os.listdir(parent_output_dir):
        abs_file_path = os.path.join(parent_output_dir, file)
        if os.path.isfile(abs_file_path) is True:
            s3_key_name = os.path.join(local_output_directory, file)
            logging.info("Uploading {} to s3 Bucket {}".format(abs_file_path, s3_key_name))
            s3_resource.Bucket(s3_bucket_name).upload_file(abs_file_path, s3_key_name)

    logging.info("finished uploading to s3, deleting local directory: {}".format(parent_output_dir))
    shutil.rmtree(parent_output_dir)
