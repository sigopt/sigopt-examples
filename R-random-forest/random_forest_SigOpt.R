# install packages
install.packages("devtools", repos = "http://cran.us.r-project.org")
library(devtools)
install_github("sigopt/SigOptR")
library(SigOptR)
install.packages('randomForest', repos = "http://cran.us.r-project.org")
library(randomForest)

# set SigOpt client token
# get your client API token here : https://sigopt.com/user/profile
Sys.setenv(SIGOPT_API_TOKEN="INSERT_YOUR_TOKEN_HERE")

# load dataset
# we are using iris dataset as an example
library(datasets)
ind <- sample(2,nrow(iris),replace=TRUE,prob=c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]
X_train <- subset(trainData, select = -c(Species))
y_train <- trainData$Species
X_test <- subset(testData, select = -c(Species))
y_test <- testData$Species

# Create an RF tuning experiment
experiment <- create_experiment(list(
  name="R RandomForest",
  parameters=list(
    list(name="mtry", type="int", bounds=list(min=1, max=ncol(iris)-1)),
    list(name="ntree", type="int", bounds=list(min=1, max=100)),
    list(name="sampsize", type="double", bounds=list(min=0.25, max=1.0)),
    list(name="nodesize", type="int", bounds=list(min=1, max=10))
  )
))

CV_accuracy <- function(assignments, X_train, y_train){
  # evaluate cross folds for accuracy
  num_folds = 5
  cv_accuracies = c()
  for (k in 1:num_folds){
    ind <- sample(2,nrow(trainData),replace=TRUE,prob=c(1-1.0/num_folds,1.0/num_folds))    
    X_cv_train = X_train[ind==1,]
    y_cv_train = y_train[ind==1]
    X_cv_valid = X_train[ind==2,]
    y_cv_valid = y_train[ind==2]
    
    rf_fit <- randomForest(y=y_cv_train,x=X_cv_train,
                           ntree=assignments$ntree,mtry=assignments$mtry,
                           sampsize=assignments$sampsize*nrow(X_cv_train),nodesize=assignments$nodesize,
                           proximity=TRUE)
    prediction <- predict(rf_fit, X_cv_valid)
    rightPred <- prediction == y_cv_valid
    accuracy <- sum(rightPred)/nrow(X_cv_valid)
    cv_accuracies <- c(accuracy, cv_accuracies)
  }
  return(c(mean(cv_accuracies),sd(cv_accuracies)))
}

best_assignment = NULL
best_val = NULL
for (i in 1:80){
  # Retrieve a suggestion
  suggestion <- create_suggestion(experiment$id)
  # Evaluate cross validated accuracy
  res <- CV_accuracy(suggestion$assignments, X_train, y_train)
  if (is.null(best_val) || res[1] > best_val  ) {
    best_val <- res[1]
    best_assignment <- suggestion$assignments
  }
  # Report observation back to SigOpt
  create_observation(experiment$id, list(suggestion=suggestion$id, value=res[1], value_stddev=res[2]))
}

# get the best observations of the experiment 
# evaluate on test set
rf_fit <- randomForest(y=y_train,x=X_train,ntree=best_assignment$ntree,
                       mtry=1, sampsize=best_assignment$sampsize*nrow(X_train),
                       nodesize=best_assignment$nodesize, proximity=TRUE)
prediction <- predict(rf_fit, X_test)
rightPred <- prediction == y_test
accuracy <- sum(rightPred)/nrow(X_test)
print(accuracy)

