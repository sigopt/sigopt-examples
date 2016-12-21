import argparse
import math
import threading

from sigopt import Connection

from data import PARAMETERS, evaluate_model

# You can find your API token at https://sigopt.com/docs/overview/authentication
SIGOPT_API_KEY = 'YOUR_API_TOKEN_HERE'

NUM_WORKERS = 2

class Worker(threading.Thread):
  def __init__(self, experiment_id):
    threading.Thread.__init__(self)
    self.experiment_id = experiment_id
    self.conn = Connection(client_token=SIGOPT_API_KEY)

  @property
  def metadata(self):
    return dict(host=threading.current_thread().name)

  @property
  def remaining_observations(self):
    experiment = self.conn.experiments(self.experiment_id).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
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

class Master(threading.Thread):
  def __init__(self):
    threading.Thread.__init__(self)
    self.conn = Connection(client_token=SIGOPT_API_KEY)
    experiment = self.conn.experiments().create(
      name='Parallel Experiment',
      parameters=PARAMETERS,
      observation_budget=len(PARAMETERS) * 20,
      metadata=dict(num_workers=NUM_WORKERS),
    )
    print("View your experiment progress: https://sigopt.com/experiment/{}".format(experiment.id))
    self.experiment_id = experiment.id

  @property
  def remaining_observations(self):
    experiment = self.conn.experiments(self.experiment_id).fetch()
    return experiment.observation_budget - experiment.progress.observation_count

  def run(self):
    tries = 3
    while (tries > 0 and self.remaining_observations > 0):
      workers = [Worker(self.experiment_id) for _ in xrange(NUM_WORKERS)]
      for worker in workers:
        worker.start()
      for worker in workers:
        worker.join()
      self.conn.experiments(self.experiment_id).suggestions().delete(state='open')
      tries -= 1

if __name__ == '__main__':
  master = Master()
  master.start()
  master.join()


