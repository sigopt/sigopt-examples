# Use SigOpt to define an experiment with constraints and optimize a version of the Adjiman function
# Learn more about SigOpt's Python Client:
# https://sigopt.com/docs/overview/python

from __future__ import print_function
from math import sin, cos

# Install sigopt with `pip install sigopt`
from sigopt import Connection

# Learn more about authenticating the SigOpt API:
# https://sigopt.com/docs/overview/authentication
conn = Connection(client_token="YOUR_SIGOPT_API_TOKEN")

# Create a SigOpt Experiment
experiment = conn.experiments().create(
  name="Adjiman Optimization with Constraints (Python)",
  project="sigopt-examples",
  parameters=[
    dict(name="x", type="double", bounds=dict(min=-1, max=2)),
    dict(name="y", type="double", bounds=dict(min=-1, max=1)),
  ],
  linear_constraints=[
    # Constraint equation: x + y >= 1
    dict(
      type="greater_than",
      threshold=1,
      terms=[
        dict(name="x", weight=1),
        dict(name="y", weight=1),
      ],
    ),
    # Constraint equation: x - y >= 1
    dict(
      type="greater_than",
      threshold=1,
      terms=[
        dict(name="x", weight=1),
        dict(name="y", weight=-1),
      ],
    ),
  ],
  observation_budget=30,
)

print("Created experiment: https://sigopt.com/experiment/{}".format(experiment.id))

# Constrained variation on the Adjiman Function http://benchmarkfcns.xyz/benchmarkfcns/adjimanfcn.html
def adjiman_function(assignments):
  x = assignments["x"]
  y = assignments["y"]
  # Multiply by -1 because SigOpt maximizes functions
  return -1 * (cos(x) * sin(y) - x / (y ** 2 + 1))

# Run the Optimization Loop between 10x - 20x the number of parameters
for _ in range(experiment.observation_budget):
  # Receive a Suggestion from SigOpt
  suggestion = conn.experiments(experiment.id).suggestions().create()

  # Evaluate the function
  value = adjiman_function(suggestion.assignments)

  # Report an Observation back to SigOpt
  conn.experiments(experiment.id).observations().create(
    suggestion=suggestion.id,
    value=value,
  )
