import argparse
import threading
import time

import sigopt

def evaluate_metric(assignments):
  # Implement this to start optimizing. Assignments is a dict-like object that maps
  # parameter names to values. Use those values to compute your metric value and return it.
  raise NotImplementedError("Add your custom function to the `evaluate_metric` function.")

class Runner(threading.Thread):
  def __init__(self, client_token, experiment_id):
    threading.Thread.__init__(self)
    self.experiment_id = experiment_id
    self.conn = sigopt.Connection(client_token=client_token)

  def run(self):
    for i in xrange(5):
      suggestion = self.conn.experiments(self.experiment_id).suggestions().create()
      value = evaluate_metric(suggestion.assignments)
      self.conn.experiments(self.experiment_id).observations().create(suggestion=suggestion.id, value=value)

def parallel_example(client_token, experiment_id, count=2):
  runners = [Runner(client_token, experiment_id) for _ in xrange(count)]
  for runner in runners:
    runner.start()
  for runner in runners:
    runner.join()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_id', type=int, required=True)
  parser.add_argument('--client_token', type=str, required=True)
  the_args = parser.parse_args()

  parallel_example(the_args.client_token, the_args.experiment_id)
