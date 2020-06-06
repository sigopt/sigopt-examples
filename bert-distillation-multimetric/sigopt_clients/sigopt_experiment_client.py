import logging


class SigOptExperiment:

    def __init__(self, connection):
        self.connection = connection

    def initialize_random_experiment(self, experiment_name, project_name, parameters_list, metrics_list, observation_budget,
                                     metadata, parallel_bandwidth=1):
        return self.initialize_experiment(experiment_name, project_name, parameters_list, list(), list(), metrics_list,
                                          observation_budget, metadata, "random", parallel_bandwidth)

    def initialize_bayesian_experiment(self, experiment_name, project_name, parameters_list, metrics_list, observation_budget, metadata, parallel_bandwidth):
        return self.initialize_experiment(experiment_name, project_name, parameters_list, list(), list(), metrics_list,
                                          observation_budget,
                                          metadata, "offline", parallel_bandwidth)

    def initialize_experiment(self, experiment_name, project_name, parameters_list, conditionals_list,
                              linear_constraints_list, metrics_list,
                              observation_budget, metadata, experiment_type, parallel_bandwidth=1):
        experiment = self.connection.experiments().create(
            name=experiment_name,
            project=project_name,
            # Define which parameters you would like to tune
            parameters=parameters_list,
            linear_constraints=linear_constraints_list,
            conditionals=conditionals_list,
            metrics=metrics_list,
            parallel_bandwidth=parallel_bandwidth,
            # Define an Observation Budget for your experiment
            observation_budget=observation_budget,
            metadata=metadata,
            type=experiment_type
        )
        logging.info("Created experiment: https://sigopt.com/experiment/%s", experiment.id)

        return experiment

    def get_initialized_experiment(self, experiment_id):
        return self.connection.experiments(experiment_id).fetch()

    def get_suggestions(self, experiment):
        sugg = self.connection.experiments(experiment.id).suggestions().create()
        sugg.assignments  = dict(sugg.assignments)
        sugg.assignments.update({
             "adam_epsilon": -11.512925464970229,
             "alpha_ce": 0.6977594228111134,
             "beta_1": 0.7484889259006229,
             "beta_2": 0.9999,
             "initializer_range": 0.13239415454956469,
             "per_compute_eval_batch_size": 25,
             "per_compute_train_batch_size": 4,
             "pruning_seed": 11,
             "temperature": 1,
             "warm_up_steps": 49,
             "weight_decay": -13.313625531829498
        })
        return sugg

    def get_suggestions_meatadata(self, experiment, metadata_dict):
        return self.connection.experiments(experiment.id).suggestions().create(metadata=metadata_dict)

    def get_best_suggestions(self, experiment):
        return self.connection.experiments(experiment.id).best_assignments().fetch()

    def update_suggestion(self, experiment_id, suggestion_id, metadata_dict):
        return self.connection.experiments(experiment_id).suggestions(suggestion_id).update(
            metadata=metadata_dict
        )

    def update_experiment(self, experiment, suggestion, evaluated_value):
        observation = self.connection.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=evaluated_value)
        return self.connection.experiments(experiment.id).fetch(), observation

    def update_experiment_multimetric_metadata(self, experiment, suggestion, evaluated_value, metadata_dict, failed=False):
        logging.info("updating experiment %s with metadata %s", experiment.id, str(metadata_dict))
        self.connection.experiments(experiment.id).observations().create(suggestion=suggestion.id,
                                                                                       values=evaluated_value,
                                                                                       failed=failed,
                                                                                       metadata=metadata_dict)
        return self.connection.experiments(experiment.id).fetch()

    def update_experiment_multimetric(self, experiment, suggestion, evaluated_values, failed=False):
        self.connection.experiments(experiment.id).observations().create(suggestion=suggestion.id,
                                                                         values=evaluated_values,
                                                                         failed=failed)
        return self.connection.experiments(experiment.id).fetch()

    def create_experiment_metadata(self, experiment, metadata_dict):
        self.connection.experiments(experiment.id).observations().create(metadata=metadata_dict)
        return self.connection.experiments(experiment.id).fetch()

    def create_observation_metadata(self, experiment, observation, metadata_dict):
        updated_observation = self.connection.experiments(experiment.id).observations(observation.id).update(metadata=metadata_dict)
        return self.connection.experiments(experiment.id).fetch(), updated_observation

    def get_all_experiments(self):
        return self.connection.experiments().fetch()

    def get_all_observations(self, experiment):
        return self.connection.experiments(experiment.id).observations().fetch()

    def archive_experiment(self, experiment):
        logging.info("archiving experiment with id: %s", experiment.id)
        self.connection.experiments(experiment.id).delete()
