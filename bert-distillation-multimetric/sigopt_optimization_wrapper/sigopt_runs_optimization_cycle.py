from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import DistillationHyperparameter
import logging
from sigopt_optimization_wrapper.sigopt_multimetric_definition import get_metrics_list
from sigopt_optimization_wrapper import runs_optimize_squad_distillation

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class SigOptCycle(object):

    RUNS_NAME = "Distillation Run"

    def __init__(self, sigopt_experiment_client, observation_budget, project_name):
        self.sigopt_experiment_client = sigopt_experiment_client
        self.observation_budget = observation_budget
        self.project_name = project_name

    def create_sigopt_experiment(self, experiment_name, hyperparameter_definition, metadata):
        sigopt_experiment = self.sigopt_experiment_client.initialize_bayesian_experiment(
            experiment_name=experiment_name,
            project_name=self.project_name,
            parameters_list=hyperparameter_definition,
            metrics_list=get_metrics_list(),
            observation_budget=self.observation_budget,
            metadata=metadata,
            parallel_bandwidth=1,
        )
        return sigopt_experiment

    def run_optimization_cycle(self, sigopt_experiment, run_parameters):
        while sigopt_experiment.progress.observation_count < sigopt_experiment.observation_budget:
            logging.info("On observation count {} of {}".format(sigopt_experiment.progress.observation_count,
                                                                sigopt_experiment.observation_budget))
            # get suggestion
            suggestion = self.sigopt_experiment_client.get_suggestions(sigopt_experiment)
            # run fine-tuning DistilBert on Squad 2.0
            parameter_values = suggestion.assignments
            # setting flag to track runs
            logging.info("running distillation training process on SQUAD 2.0")
            model, evaluated_values, failed, error_str = runs_optimize_squad_distillation.main(
                run_parameters,
                config_dict=parameter_values,
                sigopt_experiment_id=sigopt_experiment.id,
                suggestion_id=suggestion.id)

            # update experiment with metric values
            sigopt_experiment = self.sigopt_experiment_client.update_experiment_multimetric_metadata(experiment=sigopt_experiment,
                                                                                                     suggestion=suggestion,
                                                                                                     evaluated_value=evaluated_values,
                                                                                                     metadata_dict={
                                                                                                         "error_string":
                                                                                                         error_str,
                                                                                                     DistillationHyperparameter.ALPHA_SQUAD.value:
                                                                                                         parameter_values[DistillationHyperparameter.ALPHA_SQUAD.value],
                                                                                                     },
                                                                                                     failed=failed)

        return sigopt_experiment
