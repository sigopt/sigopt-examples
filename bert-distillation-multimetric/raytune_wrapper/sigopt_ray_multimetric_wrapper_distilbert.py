import copy
import logging
import pickle
from ray.tune.suggest.suggestion import SuggestionAlgorithm
import socket

logger = logging.getLogger(__name__)


class SigOptMultimetricSearch(SuggestionAlgorithm):

    """A wrapper around SigOpt to provide trial suggestions.

    Requires SigOpt to be installed. Requires user to store their SigOpt
    API key locally as an environment variable at `SIGOPT_KEY`.

    Parameters:
        hyperparameter_definition (list of dict): SigOpt configuration. Parameters will be sampled
            from this configuration and will be used to override
            parameters generated in the variant generation process.
        experiment_name (str): Name of experiment. Required by SigOpt.
        project (str): Name of project. Required by SigOpt.
        max_concurrent (int): Number of maximum concurrent trials supported
            based on the user's SigOpt plan. Defaults to 1.
        metric_name_list (list): List of metric names for the SigOpt process. This includes opitmized and stored
        metrics.
        mode (str): One of {"minimize", "maximize"}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        metric_strategy (str): One of {"store", "optimize"}. Determines whether objective should be tracked or
        optimized.
        observation_budget (int): Number of observations/trials to run SigOpt optimization loop.
    """

    def __init__(
        self,
        sigopt_experiment_client,
        project_name,
        experiment_name,
        max_concurrent,
        parallel_bandwidth,
        sigopt_experiment_id,
        metric_name_list,
        metrics_list,
        hyperparameter_definition,
        observation_budget,
        **kwargs
    ):

        self.sigopt_experiment_client = sigopt_experiment_client

        assert type(max_concurrent) is int and max_concurrent > 0

        self._max_concurrent = max_concurrent
        self._metric_list = metric_name_list
        self._live_trial_mapping = {}

        # Create SigOpt experiment with given metric name, mode, and strategy
        if sigopt_experiment_id is not None:
            logger.info("sigopt experiment provided. fetching existing experiment")
            self.experiment = self.sigopt_experiment_client.get_initialized_experiment(sigopt_experiment_id)
        else:
            logger.info("creating new sigopt experiment")
            self.experiment = self.sigopt_experiment_client.initialize_bayesian_experiment(
                experiment_name=experiment_name,
                project_name=project_name,
                parameters_list=hyperparameter_definition,
                metrics_list=metrics_list,
                observation_budget=observation_budget,
                metadata=None,
                parallel_bandwidth=parallel_bandwidth,
            )

        super(SigOptMultimetricSearch, self).__init__(**kwargs)

    def suggest(self, trial_id):
        if self._num_live_trials() >= self._max_concurrent:
            return None

        # Get new suggestion from SigOpt
        logging.info(
            "On observation count {} of {}".format(
                self.experiment.progress.observation_count, self.experiment.observation_budget
            )
        )
        suggestion = self.sigopt_experiment_client.get_suggestions(self.experiment)
        suggestion = self.sigopt_experiment_client.update_suggestion(
            experiment_id=self.experiment.id,
            suggestion_id=suggestion.id,
            metadata_dict=dict(trial_id=trial_id, host=socket.gethostname()),
        )

        self._live_trial_mapping[trial_id] = suggestion

        # mutating suggestion assignments to include id
        suggestion_assignments = suggestion.assignments
        suggestion_assignments["suggestion_id"] = suggestion.id

        return copy.deepcopy(suggestion_assignments)

    def on_trial_result(self, trial_id, result):
        pass

    def on_trial_complete(self, trial_id, result=None, error=False, early_terminated=False):
        """Notification for the completion of trial.

        If a trial fails, it will be reported as a failed Observation, telling
        the optimizer that the Suggestion led to a metric failure, which
        updates the feasible region and improves parameter recommendation.

        Creates SigOpt Observation object for trial.
        """
        if result:
            failed = result["failed"]
            if failed is True:
                metric_results = None
            else:
                metric_results = []
                for metric_name in self._metric_list:
                    metric_result = dict(name=metric_name, value=result[metric_name])
                    metric_results.append(metric_result)
            # Update the experiment object
            self.experiment = self.sigopt_experiment_client.get_initialized_experiment(self.experiment.id)
        elif error or early_terminated:
            # Reports a failed Observation
            pass

        del self._live_trial_mapping[trial_id]

    def _num_live_trials(self):
        return len(self._live_trial_mapping)

    def save(self, checkpoint_dir):
        trials_object = (self.conn, self.experiment)
        with open(checkpoint_dir, "wb") as outputFile:
            pickle.dump(trials_object, outputFile)

    def restore(self, checkpoint_dir):
        with open(checkpoint_dir, "rb") as inputFile:
            trials_object = pickle.load(inputFile)
        self.conn = trials_object[0]
        self.experiment = trials_object[1]
