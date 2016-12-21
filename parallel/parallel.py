# Learn best practices for running SigOpt in a distributed environment
#
# Learn more about SigOpt's Python Client:
# https://sigopt.com/docs/overview/python
#
import threading
from sigopt import Connection

# Define the experiment you want to create, and how to evaluate suggested points.
# For convenience we have included an example
from data import PARAMETERS, evaluate_model

# You can find your API token at https://sigopt.com/docs/overview/authentication
SIGOPT_API_KEY = 'YOUR_API_TOKEN_HERE'

NUM_WORKERS = 2

class Master(threading.Thread):
  """
  Shows what a master machine does when running SigOpt in a distributed setting.
  """

  def __init__(self):
    """
    Initialize the master thread, creating the SigOpt API connection and the experiment.
    We use the observation_budget field on the experiment to keep track of approximately
    how many total Observations we want to report. We recommend using a number between 10-20x
    the number of parameters in an experiment.
    """
    threading.Thread.__init__(self)
    self.conn = Connection(client_token=SIGOPT_API_KEY)
    experiment = self.conn.experiments().create(
      name='Parallel Experiment',
      parameters=PARAMETERS,
      observation_budget=len(PARAMETERS) * 20,
    )
    print("View your experiment progress: https://sigopt.com/experiment/{}".format(experiment.id))
    self.experiment_id = experiment.id

  @property
  def remaining_observations(self):
    """
    Re-fetch the experiment and calculate how many Observations we need to run until
    we reach the observation budget
    """
    experiment = self.conn.experiments(self.experiment_id).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
    """
    Attempt to run NUM_WORKERS worker machines. If any machines fail, retry up to
    three times, deleting openSuggestions before proceeding.
    """
    tries = 3
    while (tries > 0 and self.remaining_observations > 0):
      workers = [Worker(self.experiment_id) for _ in xrange(NUM_WORKERS)]
      for worker in workers:
        worker.start()
      for worker in workers:
        worker.join()
      self.conn.experiments(self.experiment_id).suggestions().delete(state='open')
      tries -= 1

class Worker(threading.Thread):
  """
  Shows what a worker machine does when running SigOpt in a distributed setting.
  """

  def __init__(self, experiment_id):
    """
    Initialize a worker thread, creating the SigOpt API connection and storing the previously
    created experiment's id
    """
    threading.Thread.__init__(self)
    self.experiment_id = experiment_id
    self.conn = Connection(client_token=SIGOPT_API_KEY)

  @property
  def metadata(self):
    """
    Use metadata to keep track of the host that each Suggestion and Observation is created on.
    Learn more: https://sigopt.com/docs/overview/metadata
    """
    return dict(host=threading.current_thread().name)

  @property
  def remaining_observations(self):
    """
    Re-fetch the experiment and calculate how many Observations we need to run until
    we reach the observation budget
    """
    experiment = self.conn.experiments(self.experiment_id).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
    """
    SigOpt acts as the scheduler for the Suggestions, so all you need to do is run the
    optimization loop until there are no remaining Observations to be reported.
    We handle exceptions by reporting failed Observations. Learn more about handling
    failure cases: https://sigopt.com/docs/overview/metric_failure
    """
    while self.remaining_observations > 0:
      suggestion = self.conn.experiments(self.experiment_id).suggestions().create(metadata=self.metadata)
      try:
        value = evaluate_model(suggestion.assignments)
        failed = False
      except Exception:
        value = None
        failed = True
      self.conn.experiments(self.experiment_id).observations().create(
        suggestion=suggestion.id,
        value=value,
        failed=failed,
        metadata=self.metadata,
      )

if __name__ == '__main__':
  master = Master()
  master.start()
  master.join()


