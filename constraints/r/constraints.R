# Use SigOpt to define an experiment with constraints and optimize a version of the Adjiman function
# Learn more about SigOpt's R Client:
# https://sigopt.com/docs/overview/r

# Install Packages
install.packages("SigOptR", repos = "http://cran.us.r-project.org")
library(SigOptR)

# Learn more about authenticating the SigOpt API:
# https://sigopt.com/docs/overview/authentication
Sys.setenv(SIGOPT_API_TOKEN="YOUR_SIGOPT_API_TOKEN")

# Create a SigOpt Experiment
experiment <- create_experiment(list(
  name="Adjiman Optimization with Constraints (R)",
  parameters=list(
    list(name="x", bounds=list(min=-1, max=2), type="double"),
    list(name="y", bounds=list(min=-1, max=1), type="double")
  ),
  linear_constraints=list(
    # Constraint equation: x + y >= 1
    list(
      type="greater_than",
      threshold=1,
      terms=list(
        list(name="x", weight=1),
        list(name="y", weight=1)
      )
    ),
    # Constraint equation: x - y >= 1
    list(
      type="greater_than",
      threshold=1,
      terms=list(
        list(name="x", weight=1),
        list(name="y", weight=-1)
      )
    )
  ),
  observation_budget=30
))

print(paste("Created experiment: https://sigopt.com/experiment", experiment$id, sep="/"))

# Constrained variation on the Adjiman Function http://benchmarkfcns.xyz/benchmarkfcns/adjimanfcn.html
adjiman_function <- function(assignments) {
  x = assignments$x
  y = assignments$y
  # Multiply by -1 because SigOpt maximizes functions
  return(-1 * (cos(x) * sin(y) - x / (y ** 2 + 1)))
}

# Run the Optimization Loop between 10x - 20x the number of parameters
for (i in 1:experiment$observation_budget) {
  # Receive a Suggestion from SigOpt
  suggestion <- create_suggestion(experiment$id)

  # Evaluate the function
  value <- adjiman_function(suggestion$assignments)

  # Report an Observation back to SigOpt
  create_observation(experiment$id, list(suggestion=suggestion$id, value=value))
}
