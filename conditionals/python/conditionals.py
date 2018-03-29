# Use SigOpt to define an experiment with conditionals and optimize a multivariate Gaussian distribution
# Learn more about SigOpt's Python Client:
# https://sigopt.com/docs/overview/python

from __future__ import print_function
from math import exp

# Install sigopt with `pip install sigopt`
from sigopt import Connection

# Learn more about authenticating the SigOpt API:
# https://sigopt.com/docs/overview/authentication
conn = Connection(client_token="YOUR_SIGOPT_API_TOKEN")

# Create a SigOpt Experiment
experiment = conn.experiments().create(
  name="Multivariate Gaussian distribution Optimization with Conditionals (Python)",
  conditionals=[
    dict(
      name='gaussian',
      values=['gaussian1', 'gaussian2'],
    )
  ],
  parameters=[
    dict(
      name='x',
      type='double',
      bounds=dict(min=0, max=1),
      conditions=dict(gaussian=['gaussian1']),
    ),
    dict(
      name='y',
      type='double',
      bounds=dict(min=0, max=1),
      conditions=dict(gaussian=['gaussian2']),
    ),
    dict(
      name='z',
      type='double',
      bounds=dict(min=0, max=1),
      conditions=dict(gaussian=['gaussian1', 'gaussian2']),
    )
  ],
  observation_budget=45,
)

print("Created experiment: https://sigopt.com/experiment/{}".format(experiment.id))

# Multivariate Gaussian distribution https://en.wikipedia.org/wiki/Multivariate_normal_distribution
def multivariate_gaussian_distribution(assignments):
  gaussian = assignments['gaussian']
  if gaussian == 'gaussian1':
    x = assignments['x']
    z = assignments['z']
    return .5 * exp(-10 * (.8 * (x - .2) ** 2 + .7 * (z - .5) ** 2))
  elif gaussian == 'gaussian2':
    y = assignments['y']
    z = assignments['z']
    return .5 * exp(-10 * (.7 * (y - .4) ** 2 + .3 * (z - .7) ** 2))

# Run the Optimization Loop between 10x - 20x the number of parameters
for _ in range(experiment.observation_budget):
  # Receive a Suggestion from SigOpt
  suggestion = conn.experiments(experiment.id).suggestions().create()

  # Evaluate the function
  value = multivariate_gaussian_distribution(suggestion.assignments)

  # Report an Observation back to SigOpt
  conn.experiments(experiment.id).observations().create(
    suggestion=suggestion.id,
    value=value,
  )
