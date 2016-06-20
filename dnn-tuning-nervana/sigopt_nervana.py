import time
import subprocess
import tarfile
import argparse
import io
import os
import zipfile

from ncloud.config import Config
from ncloud.commands.show import ShowModel
from ncloud.commands.train import TrainModel
from ncloud.util.api_call import api_call
from sigopt_creds import client_token as CLIENT_TOKEN

import sigopt.interface

from constant import TUNABLE_PARAMS, PAPER_PARAMS

client_token = CLIENT_TOKEN


class JobRunner(object):
  def __init__(self, num_sigopt_suggestions=300, output_file='data/sigopt.txt'):
    self.output_file = output_file
    self.num_sigopt_suggestions = num_sigopt_suggestions

  def create_experiment(self):
    """Create a SigOpt experiment for optimizing the classifier hyperparameters."""
    conn = sigopt.interface.Connection(client_token=client_token)
    experiment = conn.experiments().create(
      name="Nervana POC - {num}".format(num=int(time.time())),
      parameters=TUNABLE_PARAMS
    )
    return experiment

  def sigopt_generator(self, experiment_id):
    """Generate optimal parameter configurations using SigOpt."""
    for _ in xrange(self.num_sigopt_suggestions):
      conn = sigopt.interface.Connection(client_token=client_token)
      suggestion = conn.experiments(experiment_id).suggestions().create()
      yield suggestion

  def output_score(self, experiment_id, assignments, suggestion_id, obj, obj_std):
    """Log the score, optionally save it to file, and/or report it back to SigOpt."""
    output = "output: {suggestion} = {obj} +/- {obj_std}\n".format(
      suggestion=str(assignments),
      obj=obj,
      obj_std=obj_std,
      )
    print(output)

    conn = sigopt.interface.Connection(client_token=client_token)
    conn.experiments(experiment_id).observations().create(
        suggestion=suggestion_id,
        value=obj,
        value_stddev=0.002,
      )

  def poll_ncloud(self, active_job_ids):
    jobs_done = []
    for job_id, assignments in active_job_ids.items():
      output = ShowModel.call(Config(), job_id)
      status = output['status']
      print("job {0}: {1}".format(job_id, status))
      if status == 'Completed':
        jobs_done.append(job_id)
      elif status == 'Error':
        print "Error in job {0}, retrying...".format(job_id)
        new_job_id = self.fire_ncloud(assignments)
        active_job_ids[new_job_id] = assignments
        active_job_ids.pop(job_id)
    return jobs_done

  def get_obj_from_job_id(self, job_id):
    log = ShowModel.call(Config(), job_id, neon_log=True)
    results_path = os.path.join('/models/', str(job_id), "results")
    vals = {"format": "zip", "filter": ["*.log"]}
    zipfiles = api_call(Config(), results_path, params=vals)
    zipbytes = io.BytesIO(zipfiles)
    archive = zipfile.ZipFile(zipbytes)
    log = archive.read('neon.log')
    missclass = float(log.split('neon.callbacks.callbacks - INFO - Top1Misclass:')[-1].strip())
    return 1.0 - missclass

  @staticmethod
  def ncloud_train_from_assignments(assignments):
    temp_assignments = assignments.copy()
    args = "--learning_rate {0}".format(10.0**temp_assignments.pop('log(learning_rate)'))
    args += " --weight_decay {0}".format(10.0**temp_assignments.pop('log(weight_decay)'))
    for k, v in temp_assignments.iteritems():
      args += " --{k} {v}".format(k=k, v=v)
    return TrainModel.call(Config(), 'cifar10_allcnn_newargs.py', args=args)
  
  def fire_ncloud(self, assignments):
    output = self.ncloud_train_from_assignments(assignments)
    time.sleep(2)
    return output['id']

  def run_example(self, experiment_id=None):
    """Test various hyperparameter configurations against the dataset given a generator."""
    if experiment_id is None:
      experiment = self.create_experiment()
      experiment_id = experiment.id
    generator = self.sigopt_generator(experiment_id)
    print "Experiment id: {0}".format(experiment_id)
    active_job_ids = {}
    suggestion_ids = {}
    active_job_ids[self.fire_ncloud(PAPER_PARAMS)] = PAPER_PARAMS
    for i, suggestion in enumerate(generator):
      job_id = self.fire_ncloud(suggestion.assignments.to_json())
      suggestion_ids[job_id] = suggestion.id
      active_job_ids[job_id] = suggestion.assignments.to_json()
      if i == 2:
        break
      
    while True:
      time.sleep(60)
      print("Polling...")
      jobs_done = self.poll_ncloud(active_job_ids)
      for job_id in [j for j in jobs_done if j in active_job_ids and j in suggestion_ids]:
        obj = self.get_obj_from_job_id(job_id)
        print("Job {0} done, obj: {1}".format(job_id, obj))
        self.output_score(experiment_id, active_job_ids[job_id], suggestion_ids[job_id], obj, 0.001)
        active_job_ids.pop(job_id)
        suggestion = generator.next()
        if suggestion:
          new_job_id = self.fire_ncloud(suggestion.assignments.to_json())
          suggestion_ids[new_job_id] = suggestion.id
          active_job_ids[new_job_id] = suggestion.assignments.to_json()
      if len(active_job_ids.keys()) == 0:
        break
    print("Done.")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run Nervana POC.')
  parser.add_argument("--experiment_id", default=None, help="SigOpt experiment ID (for restarting from saved experiment)")
  args = parser.parse_args()
  runner = JobRunner()
  print('Running SigOpt...')
  runner.run_example(experiment_id=args.experiment_id)
