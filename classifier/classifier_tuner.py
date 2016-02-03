"""Class for searching for the best classification hyperparameters for a given dataset."""
from __future__ import print_function
import argparse
import itertools
import json
import numpy
import sys

from sklearn import datasets, svm, ensemble
from sigopt.interface import Connection
from sigopt.exception import ApiException
from sigopt_creds import client_token

from constant import CLASSIFIER_TYPE_TO_PARAMS, NUM_SIGOPT_SUGGESTIONS, GRID_SEARCH_WIDTH, NUM_RANDOM_SEARCHES, Dataset


class ExampleRunner(object):
    """Searches for the best classification hyperparameters for a given dataset.

    Can use the following methods for hyperparameter optimization
        - Bayesian Optimization (via SigOpt https://sigopt.com)
        - Grid Search
        - Random Search

    Example for two classifier types (with more soon):
        - 'GBC': Gradient Boosting Classifier
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        - 'SVC': Support Vector Classifier
            http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        - 'RFC': Random Forest Classifier
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Examples:
        From python:
        >>> from classifier_tuner import ExampleRunner
        >>> runner = ExampleRunner(classifier_type='GBC', ...)
        >>> runner.run_example(runner.sigopt_generator, sigopt_post=True, output_file='data/GBC_sigopt.txt')
        >>> runner.run_example(runner.grid_generator, output_file='data/GBC_grid.txt')
        >>> runner.run_example(runner.random_generator, output_file='data/GBC_random.txt')

        From a shell:
        $ python classifier_tuner.py --help

    Questions? Comments? Email contact@sigopt.com, we're happy to help!

    """

    def __init__(self, **kwargs):
        self.classifier_type = kwargs.get('classifier_type') or 'GBC'
        if self.classifier_type not in CLASSIFIER_TYPE_TO_PARAMS.keys():
          raise Exception("classifier_type must be one of %s" % CLASSIFIER_TYPE_TO_PARAMS.keys())

        self.client_token = client_token
        self.dataset_name = kwargs.get('dataset_name')
        self.test_set_size = kwargs.get('test_set_size')

        self.num_sigopt_suggestions = kwargs.get('num_sigopt_suggestions') or NUM_SIGOPT_SUGGESTIONS
        self.grid_search_width = kwargs.get('grid_search_width') or GRID_SEARCH_WIDTH
        self.num_random_searches = kwargs.get('num_random_searches') or NUM_RANDOM_SEARCHES

        self.dataset = self._load_dataset()

    def _load_dataset(self):
        """Return a Dataset with training and test data.

        Replace this with your dataset, or try one of the many public datasets at http://scikit-learn.org/stable/datasets/

        """
        print('Downloading dataset...')
        if self.dataset_name:
          if not self.test_set_size:
            raise Exception("Must provide `test_set_size` argument when using custom dataset")
          data = datasets.fetch_mldata(self.dataset_name)
          test_set_size = self.test_set_size
        else:
          # This is a small dataset that will run quickly, but is too small to provide interesting results
          data = datasets.load_digits()
          test_set_size = self.test_set_size or 300

        return Dataset(
                data['data'][:-test_set_size],
                data['target'][:-test_set_size],
                data['data'][-test_set_size:],
                data['target'][-test_set_size:],
                )

    def get_classifier(self, parameters):
        """Return a sklearn classifier with the given parameters."""
        # json unicode needs to be transformed into strings for sklearn
        parameters = dict((
          (key, str(value) if isinstance(value, unicode) else value) for key, value in parameters.iteritems()
          ))

        if self.classifier_type == 'SVC':
            return svm.SVC(**parameters)
        elif self.classifier_type == 'GBC':
            return ensemble.GradientBoostingClassifier(**parameters)
        elif self.classifier_type == 'RFC':
            return ensemble.RandomForestClassifier(n_jobs=-1, **parameters)
        else:
            raise(NotImplementedError)

    def create_experiment(self):
        """Create a SigOpt experiment for optimizing the classifier hyperparameters."""
        conn = Connection(client_token=self.client_token)
        params = CLASSIFIER_TYPE_TO_PARAMS[self.classifier_type]
        try:
            return conn.experiments().create(
                name="Example Classifier",
                parameters=params,
            )
        except ApiException as e:
            if e.status_code == 403 and 'support@sigopt.com' in str(e):
                existing_experiments = conn.experiments().fetch().data
                if existing_experiments:
                    raise Exception(
                        "You have existing experiments on sigopt.com: {0}."
                        " You have exceeded the number of experiments that can be created under your plan."
                        " Please visit https://sigopt.com/pricing to learn about plans."
                        .format(['https://sigopt.com/experiment/{0}'.format(e.id) for e in existing_experiments])
                    )
            raise

    def sigopt_generator(self, experiment):
        """Generate optimal parameter configurations using SigOpt."""
        for _ in xrange(NUM_SIGOPT_SUGGESTIONS):
            conn = Connection(client_token=self.client_token)
            suggestion = conn.experiments(experiment.id).suggestions().create()
            yield suggestion.assignments.to_json()

    def random_generator(self, experiment):
        """Return a random parameter configuration within the bounds of the parameters"""
        for _ in xrange(NUM_RANDOM_SEARCHES):
            suggestion = {}
            for param in experiment.parameters:
                if param.type == 'int':
                    suggestion[param.name] = numpy.random.randint(
                            param.bounds.min,
                            param.bounds.max,
                            )
                if param.type == 'double':
                    suggestion[param.name] = numpy.random.uniform(
                            param.bounds.min,
                            param.bounds.max,
                            )
                elif param.type == 'categorical':
                    categories = [str(cat.name) for cat in param.categorical_values]
                    suggestion[param.name] = str(numpy.random.choice(categories))

            yield suggestion

    def grid_generator(self, experiment):
        """Iterate through a grid of points within the bounds of the parameters."""
        param_value_lists = []
        for param in experiment.parameters:
            if param.type == 'int':
                param_value_lists.append(list(numpy.unique(numpy.linspace(
                    param.bounds.min,
                    param.bounds.max,
                    GRID_SEARCH_WIDTH,
                    dtype=numpy.int64,
                    ))))
            elif param.type == 'double':
                param_value_lists.append(list(numpy.linspace(
                    param.bounds.min,
                    param.bounds.max,
                    GRID_SEARCH_WIDTH,
                    )))
            elif param.type == 'categorical':
                categories = [cat.name for cat in param.categorical_values]
                param_value_lists.append(categories)

        for param_values in itertools.product(*param_value_lists):
            suggestion = {}
            for i, param_value in enumerate(param_values):
                if experiment.parameters[i].type == 'categorical':
                    suggestion[experiment.parameters[i].name] = str(param_value)
                else:
                    suggestion[experiment.parameters[i].name] = param_value

            yield suggestion

    def output_score(self, experiment, assignments, score, fout, sigopt_post=False):
        """Log the score, optionally save it to file, and/or report it back to SigOpt."""
        suggestion = [assignments[param.name] for param in experiment.parameters]

        output = "score: {suggestion} = {score}\n".format(suggestion=tuple(suggestion), score=score)
        print(output, end='')
        fout.write(output)

        if sigopt_post is True:
            conn = Connection(client_token=self.client_token)
            conn.experiments(experiment.id).observations().create(
                assignments=assignments,
                value=score,
            )
            conn.experiments(experiment.id).suggestions().delete()

    def calculate_objective(self, assignments):
        """Return the fit of the classifier with the given hyperparameters and the test data."""
        classifier = self.get_classifier(assignments)
        classifier.fit(self.dataset.X_train, self.dataset.y_train)
        return classifier.score(self.dataset.X_test, self.dataset.y_test)

    def run_example(self, experiment, generator, sigopt_post=False, output_file=None):
        """Test various hyperparameter configurations against the dataset given a generator."""
        with open(output_file, 'w') as fout:
            for assignments in generator(experiment):
                score = self.calculate_objective(assignments)
                self.output_score(experiment, assignments, score, fout, sigopt_post=sigopt_post)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier Tuner')
    parser.add_argument('--classifier-type', type=str, choices=CLASSIFIER_TYPE_TO_PARAMS.keys(), help='The type of classifier to use. Defaults to GBC')
    parser.add_argument('--dataset-name', type=str, help='The sklearn dataset to use. Defaults to datasets.load_digits().')
    parser.add_argument('--test-set-size', type=int, help='The number of points in the test set. The remainder of the dataset will be the test set.')
    parser.add_argument('--num-sigopt-suggestions', type=int, help='The number of suggestions to request from SigOpt.')
    parser.add_argument('--grid-search-width', type=int, help='How many grid points in each dimension to use for grid search')
    parser.add_argument('--num-random-searches', type=int, help='How many random search parameter configurations to test')
    args = vars(parser.parse_args())

    try:
      runner = ExampleRunner(**args)
      experiment = runner.create_experiment()

      print('Running SigOpt...')
      runner.run_example(
          experiment,
          runner.sigopt_generator,
          sigopt_post=True,
          output_file='data/{classifier_type}_{dataset_name}_sigopt.txt'.format(
              classifier_type=runner.classifier_type,
              dataset_name=runner.dataset_name,
              ),
          )

      print('Running Grid Search...')
      runner.run_example(
          experiment,
          runner.grid_generator,
          output_file='data/{classifier_type}_{dataset_name}_grid.txt'.format(
              classifier_type=runner.classifier_type,
              dataset_name=runner.dataset_name,
              ),
          )

      print('Running Random Search...')
      runner.run_example(
          experiment,
          runner.random_generator,
          output_file='data/{classifier_type}_{dataset_name}_random.txt'.format(
              classifier_type=runner.classifier_type,
              dataset_name=runner.dataset_name,
              ),
          )

      print('All done! Check out your experiment at https://sigopt.com/experiment/{0}'.format(experiment.id))
    except Exception as e:
        print(str(e), file=sys.stderr)
        print('Consult --help for for more information.', file=sys.stderr)
        exit(1)
