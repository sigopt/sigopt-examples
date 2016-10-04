# Use SigOpt to tune a Random Forest Classifier in R
# Learn more about SigOpt's R Client:
# https://sigopt.com/docs/overview/r

# Install packages
install.packages("devtools", repos = "http://cran.us.r-project.org")
library(devtools)
install_github("sigopt/SigOptR")
library(SigOptR)
install.packages('randomForest', repos = "http://cran.us.r-project.org")
library(randomForest)


# Learn more about authenticating the SigOpt API:
# https://sigopt.com/docs/overview/authentication
Sys.setenv(SIGOPT_API_TOKEN="INSERT_YOUR_TOKEN_HERE")


# Load dataset
# We are using the iris dataset as an example
library(datasets)
X <- subset(iris, select = -c(Species))
y <- iris$Species

# Create a SigOpt experiment for the Random Forest parameters
experiment <- create_experiment(list(
  name="Random Forest (R)",
  parameters=list(
    list(name="mtry", type="int", bounds=list(min=1, max=ncol(iris)-1)),
    list(name="ntree", type="int", bounds=list(min=1, max=100)),
    list(name="sampsize", type="double", bounds=list(min=0.25, max=1.0)),
    list(name="nodesize", type="int", bounds=list(min=1, max=10))
  )
))
print(paste("Created experiment: https://sigopt.com/experiment", experiment$id, sep="/"))

# Our object metric is the mean of cross validated accuracies
# We use cross validation to prevent overfitting
evaluate_model <- function(assignments, X, y) {
  # evaluate cross folds for accuracy
  num_folds = 5
  cv_accuracies = c()
  for (k in 1:num_folds){
    cv_split <- sample(2, nrow(iris), replace=TRUE, prob=c(1-1.0/num_folds, 1.0/num_folds))
    X_train <- X[cv_split==1, ]
    y_train <- y[cv_split==1]
    X_valid <- X[cv_split==2, ]
    y_valid <- y[cv_split==2]

    rf_fit <- randomForest(y=y_train,
                           x=X_train,
                           ntree=assignments$ntree,
                           mtry=assignments$mtry,
                           sampsize=assignments$sampsize*nrow(X_train),
                           nodesize=assignments$nodesize,
                           proximity=TRUE)
    prediction <- predict(rf_fit, X_valid)
    correct_predictions <- prediction == y_valid
    accuracy <- sum(correct_predictions)/nrow(X_valid)
    cv_accuracies <- c(accuracy, cv_accuracies)
  }
  return(c(mean(cv_accuracies), sd(cv_accuracies)))
}

# Run the Optimization Loop between 10x - 20x the number of parameters
for (i in 1:80) {
  # Receive a Suggestion from SigOpt
  suggestion <- create_suggestion(experiment$id)

  # Evaluate the model locally
  res <- evaluate_model(suggestion$assignments, X, y)

  # Report an Observation (with standard deviation) back to SigOpt
  create_observation(experiment$id, list(suggestion=suggestion$id,
                                         value=res[1],
                                         value_stddev=res[2]))
}

# Re-fetch the experiment to get the best observed value and assignments
experiment <- fetch_experiment(experiment$id)
best_assignments <- experiment$progress$best_observation$assignments

# To wrap up the Experiment, fit the RandomForest on the best assigments
# and train on all available data
rf <- randomForest(x=X,
                   y=y,
                   ntree=best_assignments$ntree,
                   mtry=best_assignments$mtry,
                   sampsize=best_assignments$sampsize*nrow(X),
                   nodesize=best_assignments$nodesize,
                   proximity=TRUE)
