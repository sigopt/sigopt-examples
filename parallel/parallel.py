import threading, time
import sigopt.interface

def evaluate_metric(assignments):
  # Implement this to start optimizing. Assignments is a dict-like object that maps
  # parameter names to values. Use those values to compute your metric value and return it.
  raise NotImplementedError

class Runner(threading.Thread):
  def __init__(self, client_token, experiment_id, worker_id):
    threading.Thread.__init__(self)
    self.experiment_id = experiment_id
    self.conn = sigopt.interface.Connection(client_token=client_token, worker_id=worker_id)

  def run(self):
    for i in xrange(5):
      assignments = self.conn.experiments(self.experiment_id).suggest().suggestion.assignments
      value = evaluate_metric(assignments)
      self.conn.experiments(self.experiment_id).report(data={'assignments': assignments, 'value': value})

def parallel_example(client_token, experiment_id, count=2):
  runners = [Runner(client_token, experiment_id, 'worker-%s' % (i+1)) for i in xrange(count)]
  for runner in runners:
    runner.start()
  for runner in runners:
    runner.join()

def release_worker(client_token, experiment_id, worker_id):
  conn = sigopt.interface.Connection(client_token=client_token)
  conn.experiments(experiment_id).releaseworker(worker_id=worker_id)

def release_old_workers(client_token, experiment_id, seconds_threshold=2*60*60):
  conn = sigopt.interface.Connection(client_token=client_token)
  existing_workers = conn.experiments(experiment_id).workers().workers
  for worker in existing_workers:
    # Check to see if any of the workers have been working for a long time,
    # and assume that means they have failed
    if int(time.time()) - worker.claimed_time > seconds_threshold:
      conn.experiments(experiment_id).releaseworker(worker_id=worker.id)

def release_all_workers(client_token, experiment_id):
  conn = sigopt.interface.Connection(client_token=client_token)
  existing_workers = conn.experiments(experiment_id).workers().workers
  for worker in existing_workers:
    conn.experiments(experiment_id).releaseworker(worker_id=worker.id)
