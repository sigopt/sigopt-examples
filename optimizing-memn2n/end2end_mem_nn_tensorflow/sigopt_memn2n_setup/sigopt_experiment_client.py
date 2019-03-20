import logging

class SigOptExperiment:

    def __init__(self, connection):
        self.connection = connection

    def initialize_random_experiment(self, experiment_name, parameters_list, metrics_list, observation_budget, metadata):
        return self.initialize_bayesian_experiment(experiment_name, parameters_list, None, metrics_list, observation_budget, metadata, "random")

    def initialize_bayesian_experiment(self, experiment_name, parameters_list, metrics_list, observation_budget, metadata):
        return self.initialize_experiment(experiment_name, parameters_list, None, metrics_list, observation_budget, metadata, "offline")

    def initialize_experiment(self, experiment_name, parameters_list, conditionals_list, metrics_list, observation_budget, metadata, experiment_type):
        experiment = self.connection.experiments().create(
            name=experiment_name,
            # Define which parameters you would like to tune
            parameters=parameters_list,
            conditionals=conditionals_list,
            metrics=metrics_list,
            parallel_bandwidth=1,
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
        return self.connection.experiments(experiment.id).suggestions().create()

    def get_best_suggestions(self, experiment):
        return self.connection.experiments(experiment.id).best_assignments().fetch()

    def update_experiment(self, experiment, suggestion, evaluated_value):
        observation = self.connection.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=evaluated_value)
        return self.connection.experiments(experiment.id).fetch(), observation

    def update_experiment_metadata(self, experiment, suggestion, evaluated_value, metadata_dict):
        logging.info("updating experiment %s with metadata %s", experiment.id, str(metadata_dict))
        observation = self.connection.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=evaluated_value, metadata=metadata_dict)
        return self.connection.experiments(experiment.id).fetch(), observation

    def update_experiment_multimetric(self, experiment, suggestion, evaluated_value):
        self.connection.experiments(experiment.id).observations().create(suggestion=suggestion.id, values=evaluated_value)
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
