# Use SigOpt to define an experiment with conditionals and optimize a multivariate Gaussian distribution
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
  name="Multivariate Gaussian distribution Optimization with Conditionals (R)",
  conditionals=list(
    list(
      name='gaussian',
      values=c('gaussian1', 'gaussian2')
    )
  ),
  parameters=list(
    list(
      name='x',
      type='double',
      bounds=list(min=0, max=1),
      conditions=list(gaussian=c('gaussian1'))
    ),
    list(
      name='y',
      type='double',
      bounds=list(min=0, max=1),
      conditions=list(gaussian=c('gaussian2'))
    ),
    list(
      name='z',
      type='double',
      bounds=list(min=0, max=1),
        conditions=list(gaussian=c('gaussian1', 'gaussian2'))
    )
  ),
  observation_budget=45
))

print(paste("Created experiment: https://sigopt.com/experiment", experiment$id, sep="/"))

# Multivariate Gaussian distribution https://en.wikipedia.org/wiki/Multivariate_normal_distribution
multivariate_gaussian_distribution <- function(assignments) {
  gaussian = assignments$gaussian
  if (gaussian == 'gaussian1') {
    x = assignments$x
    z = assignments$z
    return (.5 * exp(-10 * (.8 * (x - .2) ** 2 + .7 * (z - .5) ** 2)))
  } else if (gaussian == 'gaussian2') {
    y = assignments$y
    z = assignments$z
    return (.5 * exp(-10 * (.7 * (y - .4) ** 2 + .3 * (z - .7) ** 2)))
  }
}

# Run the Optimization Loop between 10x - 20x the number of parameters
for (i in 1:experiment$observation_budget) {
  # Receive a Suggestion from SigOpt
  suggestion <- create_suggestion(experiment$id)

  # Evaluate the function
  value <- multivariate_gaussian_distribution(suggestion$assignments)

  # Report an Observation back to SigOpt
  create_observation(experiment$id, list(suggestion=suggestion$id, value=value))
}
