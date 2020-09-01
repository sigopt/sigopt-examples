from sigopt_clients import sigopt_experiment_client
from sigopt_optimization_wrapper import sigopt_hyperparameter_definition
from sigopt import Connection
from sigopt_optimization_wrapper import sigopt_runs_optimization_cycle
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import OptimizationRunParameters, RunParameters
from squad_distillation_abstract_clis.a_optimizaton_run_squad_cli import OptimizeDistilBertQuadCLI
import logging
import os

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

SIGOPT_API_TOKEN = "SIGOPT_API_TOKEN"


class SigOptDistilBertQuadCLI(OptimizeDistilBertQuadCLI):

    def __init__(self):
        super().__init__()

    def get_commandline_args(self):
        return self.define_run_commandline_args()


if __name__ == "__main__":

    optimization_cli = SigOptDistilBertQuadCLI()

    args = optimization_cli.get_commandline_args().parse_args()
    args_dict = vars(args)
    logging.info("running optimization experiment with arguments: {}".format(args_dict))

    sigopt_client_connection = Connection()
    sigopt_client = sigopt_experiment_client.SigOptExperiment(connection=sigopt_client_connection)

    logging.info("tracking flag turned on, will execute optimization with runs tracking")
    sigopt_cycle = sigopt_runs_optimization_cycle.SigOptCycle(sigopt_experiment_client=sigopt_client,
                                                              observation_budget=args_dict[
                                                                      OptimizationRunParameters.SIGOPT_OBSERVATION_BUDGET.value],
                                                              project_name=args_dict[
                                                                      OptimizationRunParameters.PROJECT_NAME.value])

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
        if args_dict[OptimizationRunParameters.SIGOPT_EXPERIMENT_ID.value] is not None:
            sigopt_experiment = sigopt_client.get_initialized_experiment(
                args_dict[OptimizationRunParameters.SIGOPT_EXPERIMENT_ID.value])
        else:
            sigopt_experiment = sigopt_cycle.create_sigopt_experiment(
                experiment_name=args_dict[OptimizationRunParameters.EXPERIMENT_NAME.value],
                hyperparameter_definition=sigopt_hyperparameter_definition.get_sigopt_hyperparameter_list(),
                metadata=run_params_dict)
        logging.debug("SigOpt Experiment: {}".format(sigopt_experiment))
        # set output directory
        args_dict[OptimizationRunParameters.SIGOPT_RUN_DIRECTORY.value] = os.path.join(args_dict[RunParameters.OUTPUT_DIR.value],
                                                                                            str(sigopt_experiment.id))
        logging.info("Run outputs can be found: {}".format(args_dict[OptimizationRunParameters.SIGOPT_RUN_DIRECTORY.value]))
        sigopt_experiment = sigopt_cycle.run_optimization_cycle(sigopt_experiment,
                                                                run_parameters=args_dict,
                                                                )
    else:
        raise RuntimeError("Currently does not support custom HPO ranges. Change class file to accomodate.")
