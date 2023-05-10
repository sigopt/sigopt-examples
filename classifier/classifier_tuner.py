"""Class for searching for the best classification hyperparameters for a given dataset."""
from __future__ import print_function
import argparse
import itertools
import time
import numpy
import sys

from sklearn import datasets, svm, ensemble
from sigopt import Connection
from sigopt.exception import ApiException

from constant import (
  CLASSIFIER_TYPE_TO_PARAMS,
  NUM_SIGOPT_SUGGESTIONS,
  GRID_SEARCH_WIDTH,
  NUM_RANDOM_SEARCHES,
  Dataset,
)


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
    self.classifier_type = kwargs.get("classifier_type") or "GBC"
    if self.classifier_type not in CLASSIFIER_TYPE_TO_PARAMS.keys():
      raise Exception(
        "classifier_type must be one of %s" % CLASSIFIER_TYPE_TO_PARAMS.keys()
      )

    self.client_token = kwargs.get("client_token")
    self.dataset_name = kwargs.get("dataset_name")
    self.test_set_size = kwargs.get("test_set_size")

    self.num_sigopt_suggestions = (
      kwargs.get("num_sigopt_suggestions") or NUM_SIGOPT_SUGGESTIONS
    )
    self.grid_search_width = kwargs.get("grid_search_width") or GRID_SEARCH_WIDTH
    self.num_random_searches = kwargs.get("num_random_searches") or NUM_RANDOM_SEARCHES

    self.dataset = self._load_dataset()

  def _load_dataset(self):
    """Return a Dataset with training and test data.

        Replace this with your dataset, or try one of the many public datasets at http://scikit-learn.org/stable/datasets/

        """
    print("Downloading dataset...")
    if self.dataset_name:
      if not self.test_set_size:
        raise Exception(
          "Must provide `test_set_size` argument when using custom dataset"
        )
      data = datasets.fetch_mldata(self.dataset_name)
      test_set_size = self.test_set_size
    else:
      # This is a small dataset that will run quickly, but is too small to provide interesting results
      data = datasets.load_digits()
      test_set_size = self.test_set_size or 300

    return Dataset(
      data["data"][:-test_set_size],
      data["target"][:-test_set_size],
      data["data"][-test_set_size:],
      data["target"][-test_set_size:],
    )

  def get_classifier(self, parameters):
    """Return a sklearn classifier with the given parameters."""

    if self.classifier_type == "SVC":
      return svm.SVC(**parameters)
    elif self.classifier_type == "GBC":
      return ensemble.GradientBoostingClassifier(**parameters)
    elif self.classifier_type == "RFC":
      return ensemble.RandomForestClassifier(n_jobs=-1, **parameters)
    else:
      raise (NotImplementedError)

  def create_experiment(self):
    """Create a SigOpt experiment for optimizing the classifier hyperparameters."""
    if self.client_token:
      self.conn = Connection(client_token=self.client_token)
    else:
      self.conn = Connection(driver="lite")
    params = CLASSIFIER_TYPE_TO_PARAMS[self.classifier_type]
    try:
      return self.conn.experiments().create(
        name="Example Classifier",
        parameters=params,
        metrics=[dict(name="classifier_score", objective="maximize")],
        observation_budget=self.num_sigopt_suggestions,
      )
    except ApiException as err:
      if err.status_code == 403 and "support@sigopt.com" in str(err):
        existing_experiments = list(self.conn.experiments().fetch().iterate_pages())
        if existing_experiments:
          raise Exception(
            "You have existing experiments on sigopt.com: {0}."
            " You have exceeded the number of experiments that can be created under your plan."
            " Please visit https://sigopt.com/contact to upgrade your plan.".format(
              [
                "https://sigopt.com/experiment/{0}".format(e.id)
                for e in existing_experiments
              ]
            )
          )
      raise

  def sigopt_generator(self, experiment):
    """Generate optimal parameter configurations using SigOpt."""
    for _ in range(experiment.observation_budget):
      suggestion = self.conn.experiments(experiment.id).suggestions().create()
      yield suggestion.assignments.to_json()

  def random_generator(self, experiment):
    """Return a random parameter configuration within the bounds of the parameters"""
    for _ in range(self.num_random_searches):
      suggestion = {}
      for param in experiment.parameters:
        if param.type == "int":
          suggestion[param.name] = numpy.random.randint(
            param.bounds.min,
            param.bounds.max,
          )
        if param.type == "double":
          suggestion[param.name] = numpy.random.uniform(
            param.bounds.min,
            param.bounds.max,
          )
        elif param.type == "categorical":
          categories = [str(cat.name) for cat in param.categorical_values]
          suggestion[param.name] = str(numpy.random.choice(categories))

      yield suggestion

  def grid_generator(self, experiment):
    """Iterate through a grid of points within the bounds of the parameters."""
    param_value_lists = []
    for param in experiment.parameters:
      if param.type == "categorical":
        categories = [cat.name for cat in param.categorical_values]
        param_value_lists.append(categories)
      else:
        linspace = numpy.linspace(
          param.bounds.min,
          param.bounds.max,
          self.grid_search_width,
        )
        if param.type == "int":
          param_value_lists.append(
            [int(i) for i in numpy.unique([round(i) for i in linspace])]
          )
        else:
          param_value_lists.append(linspace)

    for param_values in itertools.product(*param_value_lists):
      suggestion = {}
      for i, param_value in enumerate(param_values):
        if experiment.parameters[i].type == "categorical":
          suggestion[experiment.parameters[i].name] = str(param_value)
        else:
          suggestion[experiment.parameters[i].name] = param_value

      yield suggestion

  def output_score(self, experiment, assignments, score, fout, sigopt_post=False):
    """Log the score, optionally save it to file, and/or report it back to SigOpt."""
    suggestion = [assignments[param.name] for param in experiment.parameters]

    output = "score: {suggestion} = {score}\n".format(
      suggestion=tuple(suggestion), score=score
    )
    print(output, end="")
    fout.write(output)

    if sigopt_post is True:
      self.conn.experiments(experiment.id).observations().create(
        assignments=assignments,
        values=[dict(name="classifier_score", value=score)],
      )

  def calculate_objective(self, assignments):
    """Return the fit of the classifier with the given hyperparameters and the test data."""
    classifier = self.get_classifier(assignments)
    classifier.fit(self.dataset.X_train, self.dataset.y_train)
    return classifier.score(self.dataset.X_test, self.dataset.y_test)

  def run_example(
    self, experiment, generator, sigopt_post=False, output_file=None, type=""
  ):
    """Test various hyperparameter configurations against the dataset given a generator."""
    best_score = -1000
    best_assignments = {}
    run_count = 0
    start_time = time.perf_counter()
    with open(output_file, "w") as fout:
      for assignments in generator(experiment):
        run_count += 1
        score = self.calculate_objective(assignments)
        # this will only work for one dimensional metrics that are trying to maximize
        # Sigopt will automatically handle more complicated definitions of 'best'- see
        # below about best_assignments for how to find that - but numpy grid and random
        # do not, so we use this to find "best" for those as well for comparison
        if score > best_score:
          best_score = score
          best_assignments = assignments
        self.output_score(experiment, assignments, score, fout, sigopt_post=sigopt_post)
      stop_time = time.perf_counter()
      output = f"Best score in this {type} experiment: {best_assignments} had a score of {best_score}. It took {run_count} runs, with a total time of {stop_time - start_time} seconds."
      fout.write(output)
      return output


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Classifier Tuner")
  parser.add_argument(
    "--client-token",
    type=str,
    help="Your sigopt API token. Get this from https://sigopt.com/tokens",
  )
  parser.add_argument(
    "--classifier-type",
    type=str,
    choices=CLASSIFIER_TYPE_TO_PARAMS.keys(),
    help="The type of classifier to use. Defaults to GBC.",
    default="GBC",
  )
  parser.add_argument(
    "--dataset-name",
    type=str,
    help="The sklearn dataset to use. Defaults to datasets.load_digits().",
  )
  parser.add_argument(
    "--test-set-size",
    type=int,
    help="The number of points in the test set. The remainder of the dataset will be the test set.",
  )
  parser.add_argument(
    "--num-sigopt-suggestions",
    type=int,
    help="The number of suggestions to request from SigOpt.",
    default=NUM_SIGOPT_SUGGESTIONS,
  )
  parser.add_argument(
    "--grid-search-width",
    type=int,
    help="How many grid points in each dimension to use for grid search",
    default=GRID_SEARCH_WIDTH,
  )
  parser.add_argument(
    "--num-random-searches",
    type=int,
    help="How many random search parameter configurations to test",
    default=NUM_RANDOM_SEARCHES,
  )
  args = vars(parser.parse_args())

  try:
    runner = ExampleRunner(**args)
    experiment = runner.create_experiment()

    print("Running SigOpt...")
    sigopt_output = runner.run_example(
      experiment,
      runner.sigopt_generator,
      sigopt_post=True,
      output_file="data/{classifier_type}_{dataset_name}_sigopt.txt".format(
        classifier_type=runner.classifier_type,
        dataset_name=runner.dataset_name,
      ),
      type="Sigopt",
    )

    print("Running Grid Search...")
    grid_output = runner.run_example(
      experiment,
      runner.grid_generator,
      output_file="data/{classifier_type}_{dataset_name}_grid.txt".format(
        classifier_type=runner.classifier_type,
        dataset_name=runner.dataset_name,
      ),
      type="Grid",
    )

    print("Running Random Search...")
    random_output = runner.run_example(
      experiment,
      runner.random_generator,
      output_file="data/{classifier_type}_{dataset_name}_random.txt".format(
        classifier_type=runner.classifier_type,
        dataset_name=runner.dataset_name,
      ),
      type="Random",
    )

    sigopt_best_assignments = list(
      runner.conn.experiments(experiment.id).best_assignments().fetch().iterate_pages()
    )

    print(sigopt_output)
    print(grid_output)
    print(random_output)
    print(f"All done! Sigopt calculated the best results at {sigopt_best_assignments}")
    if runner.client_token:
      print(
        "All done! Check out your experiment at https://sigopt.com/experiment/{0}".format(
          experiment.id
        )
      )
  except Exception as e:
    print(str(e), file=sys.stderr)
    print("Consult --help for for more information.", file=sys.stderr)
    exit(1)
