import argparse
import os
from subprocess import PIPE, Popen
import sys

from sigopt import Connection

class SubProcessEvaluator(object):
  def __init__(self, command):
    self.command = command

  # Take a suggestion from sigopt and evaluate your function
  # Sends command line arguments to your executable file with the same names as the
  # parameters of your experiment. Expected output is one line containing a float that
  # is your function evaluated at the suggested assignments.
  # For example, if your command is './test' and you have one double parameter with suggested
  # value 11.05, this script will run
  #         ./test --x=11.05
  def evaluate_metric(self, assignments):
    arguments = [
      '--{}={}'.format(param_name, assignment)
      for param_name, assignment
      in assignments.to_json().iteritems()
    ]
    process = Popen(self.command.split() + arguments, stdout=PIPE, stderr=PIPE)
    (stdoutdata,stderrdata) = process.communicate()
    sys.stderr.write(stderrdata)
    return float(stdoutdata.strip())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--command', required=True, help="The command to run the function whose parameters you would "
    "like to optimize. Should accept parameters as command line argument and output only the evaluated metric at the "
    "suggested point.")
  parser.add_argument('--experiment_id', required=True, help="The parameters of this experiment should be the "
    "same type and name of the command line arguments to your executable file.")
  parser.add_argument('--client_token', required=True, help="Find your CLIENT_TOKEN at https://sigopt.com/user/profile")
  the_args = parser.parse_args()

  connection = Connection(client_token=the_args.client_token)
  experiment = connection.experiments(the_args.experiment_id).fetch()
  connection.experiments(the_args.experiment_id).suggestions().delete(state="open")
  evaluator = SubProcessEvaluator(the_args.command)

  # In a loop: receive a suggestion, evaluate the metric, report an observation
  while True:
    suggestion = connection.experiments(experiment.id).suggestions().create()
    print('Evaluating at suggested assignments: {0}'.format(suggestion.assignments))
    value = evaluator.evaluate_metric(suggestion.assignments)
    print('Reporting observation of value: {0}'.format(value))
    connection.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=value,
    )
