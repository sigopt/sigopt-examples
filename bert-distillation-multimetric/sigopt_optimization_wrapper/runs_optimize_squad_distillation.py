import logging
import sigopt
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import OptimizationRunParameters
from squad_fine_tuning.optimize_squad_distillation import OptimizeSquadDistillation


class RunsOptimizeSquadDistillation(OptimizeSquadDistillation):
    def __init__(self, args_dict):
        super().__init__(args_dict)


def main(args_dict, config_dict, sigopt_experiment_id, suggestion_id):
    logging.info("arguments passed to distillation training: {}".format(args_dict))
    logging.info("configuration passed to distillation training: {}".format(config_dict))

    runs_optimize_squad_distillation = RunsOptimizeSquadDistillation(args_dict=args_dict)
    parameter_values, args_dict, run_training_squad_distillation, model = runs_optimize_squad_distillation.setup_run(
        suggestion_id, args_dict, config_dict
    )

    all_parameters = dict()
    all_parameters.update(args_dict)
    all_parameters.update(parameter_values)

    connection = sigopt.Connection()
    suggestion = connection.experiments(sigopt_experiment_id).suggestions(suggestion_id).fetch()

    with sigopt.create_run(
        name="Distillation Run_experiment_{}_suggestion_{}".format(sigopt_experiment_id, suggestion_id),
        project=args_dict[OptimizationRunParameters.PROJECT_NAME.value],
        suggestion=suggestion,
    ) as run:
        run.log_dataset("SQUAD 2.0")
        run.log_model("DistilBert for question answering")
        failed, error_str, evaluated_values, results, model = runs_optimize_squad_distillation.try_distillation_tuning(
            run_training_squad_distillation, all_parameters, model, run
        )
        if failed is True:
            run.log_failure()
            run.log_metadata(key="error_str", value=error_str)
        else:
            run.log_checkpoint(results)
            for evaluated_metric in evaluated_values:
                run.log_metric(evaluated_metric["name"], evaluated_metric["value"])
