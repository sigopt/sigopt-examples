from squad_distillation_abstract_clis.a_optimizaton_run_squad_cli import OptimizeDistilBertQuadCLI
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RayTuneRunParameters, OptimizationRunParameters, RunParameters
from sigopt_clients import sigopt_experiment_client
import ray
from ray.tune import run
from ray.tune.schedulers import FIFOScheduler
from sigopt import Connection
import logging
from raytune_wrapper import sigopt_ray_multimetric_wrapper_distilbert, ray_optimize_squad_distillation
from sigopt_optimization_wrapper import sigopt_hyperparameter_definition
from sigopt_optimization_wrapper.sigopt_multimetric_definition import get_metrics_list, get_metric_names
import os
import shutil

SIGOPT_API_TOKEN = "SIGOPT_API_TOKEN"


class RayTuneSigOptDistilBertSquadCLI(OptimizeDistilBertQuadCLI):

    RAY_OUTPUT_DIR = os.path.expanduser("~/ray_results")

    def __init__(self):
        super().__init__()

    def define_run_commandline_args(self):
        parser = super().define_run_commandline_args()
        parser.add_argument(super().parser_prefix + RayTuneRunParameters.PARALLEL.value, type=int, required=True,
                            help="Number of parallel workers to use")
        parser.add_argument(super().parser_prefix + RayTuneRunParameters.MAX_CONCURRENT.value, type=int, required=True,
                            help="Total number of current concurrent runs.")
        parser.add_argument(super().parser_prefix + RayTuneRunParameters.NUM_CPU.value, type=int, required=True,
                            help="Number of cpus per run")
        parser.add_argument(super().parser_prefix + RayTuneRunParameters.NUM_GPU.value, type=int, required=True,
                            help="Number of gpus per run")
        parser.add_argument(super().parser_prefix + RayTuneRunParameters.RAY_ADDRESS.value, type=str, required=False,
                            help="ip address for Ray cluster")
        parser.add_argument(super().parser_prefix + RayTuneRunParameters.CLEAN_RAYTUNE_OUTPUT.value,
                            action='store_true',
                            help="when set, will delete ray tune output directory")
        parser.add_argument(super().parser_prefix + RayTuneRunParameters.RAY_OUTPUT_DIRECTORY.value, type=str,
                            required=False, default=self.RAY_OUTPUT_DIR)
        return parser

    def get_commandline_args(self):
        return self.define_run_commandline_args()


def clean_up_ray_output(ray_output_directory):
    logging.info("Deleting ray tune output directory: {}".format(ray_output_directory))
    if os.path.exists(ray_output_directory):
        shutil.rmtree(ray_output_directory)


if __name__ == "__main__":
    ray_optimization_cli = RayTuneSigOptDistilBertSquadCLI()

    args = ray_optimization_cli.get_commandline_args().parse_args()
    args_dict = vars(args)
    logging.info("running optimization experiment with arguments: {}".format(args_dict))

    logging.info("connecting to Ray cluster")
    if args.ray_address is not None:
        ray.init(address=args_dict[RayTuneRunParameters.RAY_ADDRESS.value])
    else:
        ray.init()

    sigopt_client = sigopt_experiment_client.SigOptExperiment(connection=(
        Connection(client_token=args_dict[OptimizationRunParameters.API_TOKEN.value])))

    run_params_dict = dict()
    for default_param_key, default_param_value in args_dict.items():
        if type(default_param_value) == bool:
            if default_param_value is True:
                run_params_dict[default_param_key] = "True"
            else:
                run_params_dict[default_param_key] = "False"
        else:
            if default_param_key is not OptimizationRunParameters.API_TOKEN.value:
                run_params_dict[default_param_key] = default_param_value

    if args_dict[OptimizationRunParameters.USE_HPO_DEFAULT_RANGES.value] is True:
        sigopt_multimetric_search = sigopt_ray_multimetric_wrapper_distilbert.SigOptMultimetricSearch(
            sigopt_experiment_client=sigopt_client,
            project_name=args_dict[
                OptimizationRunParameters.PROJECT_NAME.value],
            experiment_name=args_dict[
                OptimizationRunParameters.EXPERIMENT_NAME.value],
            max_concurrent=args_dict[
                RayTuneRunParameters.MAX_CONCURRENT.value],
            hyperparameter_definition=sigopt_hyperparameter_definition.get_sigopt_hyperparameter_list(),
            observation_budget=args_dict[
                OptimizationRunParameters.SIGOPT_OBSERVATION_BUDGET.value],
            parallel_bandwidth=args_dict[
                RayTuneRunParameters.PARALLEL.value],
            metrics_list=get_metrics_list(),
            metric_name_list=get_metric_names(),
            sigopt_experiment_id=args_dict[
                OptimizationRunParameters.SIGOPT_EXPERIMENT_ID.value]
        )
        # set output directory
        args_dict[OptimizationRunParameters.SIGOPT_RUN_DIRECTORY.value] = os.path.join(args_dict[RunParameters.OUTPUT_DIR.value], str(sigopt_multimetric_search.experiment.id))
        args_dict[OptimizationRunParameters.SIGOPT_EXPERIMENT_ID.value] = str(sigopt_multimetric_search.experiment.id)
        logging.info(
            "Run outputs can be found: {}".format(args_dict[OptimizationRunParameters.SIGOPT_RUN_DIRECTORY.value]))
        # create config to pass arguments to ray tune cycle
        config = dict()
        config["config"] = args_dict
        ray.tune.run(ray_optimize_squad_distillation.main,
                     name=args_dict[OptimizationRunParameters.EXPERIMENT_NAME.value],
                     local_dir=args_dict[RayTuneRunParameters.RAY_OUTPUT_DIRECTORY.value],
                     search_alg=sigopt_multimetric_search,
                     num_samples=args_dict[OptimizationRunParameters.SIGOPT_OBSERVATION_BUDGET.value],
                     scheduler=FIFOScheduler(),
                     resources_per_trial=dict(cpu=args_dict[RayTuneRunParameters.NUM_CPU.value],
                                              gpu=args_dict[RayTuneRunParameters.NUM_GPU.value]),
                     **config)
        if args_dict[RayTuneRunParameters.CLEAN_RAYTUNE_OUTPUT.value] is True:
            clean_up_ray_output(os.path.join(args_dict[RayTuneRunParameters.RAY_OUTPUT_DIRECTORY.value],
                                        args_dict[OptimizationRunParameters.EXPERIMENT_NAME.value]))
    else:
        raise RuntimeError("Currently only supports default hyperparameter ranges.")
