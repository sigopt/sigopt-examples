# Use SigOpt to tune an XGBoost softmax classifier in R
# Learn more about SigOpt's R Client:
# https://sigopt.com/docs/overview/r

# Install packages
install.packages("devtools", repos = "http://cran.us.r-project.org")
library(devtools)
install_github("sigopt/SigOptR")
library(SigOptR)

install.packages("xgboost", repos = "http://cran.us.r-project.org")
library(xgboost)
library(Matrix)

# Learn more about authenticating the SigOpt API:
# https://sigopt.com/docs/overview/authentication
Sys.setenv(SIGOPT_API_TOKEN="INSERT_YOUR_TOKEN_HERE")

# Load dataset
# We are using the iris dataset as an example
library(datasets)
X <- subset(iris, select = -c(Species))
y <- iris$Species

# Construct the data matrix for XGBoost
# XGBoost uses numeric labels instead of strings - numeric labels start from 0
data <- xgb.DMatrix(data=as.matrix(X), label=as.numeric(y)-1)
# Get mapping from numeric to string labels - indexing starts at 1
levels = levels(y)
# Get number of different classes. Numeric labels range from [0, num_class)
num_class = length(levels)

# Create a SigOpt experiment for the XGBoost parameters
experiment <- create_experiment(list(
  name="XGBoost Softmax Classifier (R)",
  parameters=list(
    list(name="nrounds", type="int", bounds=list(min=10, max=100)),
    list(name="eta", type="double", bounds=list(min=0.25, max=1.0)),
    list(name="gamma", type="int", bounds=list(min=3, max=10)),
    list(name="max_depth", type="int", bounds=list(min=3, max=15)),
    list(name="subsample", type="double", bounds=list(min=0.5, max=1.0))
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
    data_train <- xgb.DMatrix(data=as.matrix(X_train), label=as.numeric(y_train)-1)

    boosted_clf <- xgb.train(
      data=data_train,
      booster="gbtree",
      objective="multi:softmax",
      num_class=num_class,
      nrounds=assignments$nrounds,
      eta=assignments$eta,
      gamma=assignments$gamma,
      max_depth=assignments$max_depth,
      subsample=assignments$subsample
    ) 
    
    prediction <- predict(boosted_clf, as.matrix(X_valid))
    # Offset the prediction to line up with levels indexing
    correct_predictions <- levels[prediction+1] == y_valid
    accuracy <- sum(correct_predictions)/nrow(X_valid)
    cv_accuracies <- c(accuracy, cv_accuracies)
  }
  return(c(mean(cv_accuracies), sd(cv_accuracies)))
}

# Run the Optimization Loop between 10x - 20x the number of parameters
for (i in 1:100) {
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

# To wrap up the Experiment, fit the model with the best assignments
# and train on all available data
boosted_clf <- xgb.train(
  data=data,
  booster="gbtree",
  objective="multi:softmax",
  num_class=num_class,
  nrounds=best_assignments$nrounds,
  eta=best_assignments$eta,
  gamma=best_assignments$gamma,
  max_depth=best_assignments$max_depth,
  subsample=best_assignments$subsample
)
