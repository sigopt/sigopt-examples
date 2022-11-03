# Use SigOpt to tune a text classifier in R
# Learn more about SigOpt's R Client:
# https://sigopt.com/docs/overview/r

install.packages("devtools", repos = "http://cran.us.r-project.org")
library(devtools)
install_github("sigopt/SigOptR")
library(SigOptR)
install.packages("rjson", repos = "http://cran.us.r-project.org")
library(rjson)
install.packages("text2vec", repos = "http://cran.us.r-project.org")
library(text2vec)
install.packages("glmnet", repos = "http://cran.us.r-project.org")
library(glmnet)

# Learn more about authenticating the SigOpt API:
# https://sigopt.com/docs/overview/authentication
Sys.setenv(SIGOPT_API_TOKEN="YOUR_API_TOKEN_HERE")

# load text training data
POSITIVE_TEXT <- fromJSON(file="https://public.sigopt.com/POSITIVE_list.json")
NEGATIVE_TEXT <- fromJSON(file="https://public.sigopt.com/NEGATIVE_list.json")

# optimization metric : see blogpost https://sigopt.com/blog/automatically-tuning-text-classifiers/
sentiment_metric <- function(POS_TEXT, NEG_TEXT, params) {
  min_ngram <- params$min_n_gram
  max_ngram <- min_ngram + params$n_gram_offset
  min_doc_freq <- exp(params$log_min_df)
  max_doc_freq <- min_doc_freq + params$df_offset
  text = c(POSITIVE_TEXT, NEGATIVE_TEXT)

  # Construct a matrix representation of the text
  it <- itoken(text, tolower, word_tokenizer)
  vocab <- create_vocabulary(it, ngram = c(min_ngram, max_ngram))
  pruned_vocab = prune_vocabulary(vocab, doc_proportion_min = min_doc_freq, doc_proportion_max = max_doc_freq)
  it <- itoken(text, tolower, word_tokenizer)
  vectorizer <- vocab_vectorizer(pruned_vocab)

  X <- create_dtm(it, vectorizer)
  y <- c(rep(1, length(POSITIVE_TEXT)), rep(0, length(NEGATIVE_TEXT)))

  # Perform cross-validation
  num_folds = 5
  cv_accuracies = c()
  for(i in 1:num_folds) {
    cv_split <- sample(2, nrow(X), replace=TRUE, prob=c(0.7, 0.3))
    X_train <- X[cv_split==1, ]
    y_train <- y[cv_split==1]
    X_valid <- X[cv_split==2, ]
    y_valid <- y[cv_split==2]

    fit <- glmnet(X_train, y_train, family = "binomial", lambda=exp(params$log_reg_coef), alpha=params$l1_coef)
    prediction <-predict(fit, X_valid, type="class")

    correct_predictions <- prediction == y_valid
    accuracy <- sum(correct_predictions)/nrow(X_valid)
    cv_accuracies <- c(accuracy, cv_accuracies)
  }

  return(c(mean(cv_accuracies), sd(cv_accuracies)))
}

experiment <- create_experiment(list(
  name="Sentiment LR Classifier (R)",
  parameters=list(
    list(name="l1_coef",       type="double", bounds=list(min=0, max=1.0)),
    list(name="log_reg_coef",  type="double", bounds=list(min=log(0.000001), max=log(100.0))),
    list(name="min_n_gram",    type="int",    bounds=list(min=1, max=2)),
    list(name="n_gram_offset", type="int",    bounds=list(min=0, max=2)),
    list(name="log_min_df",    type="double", bounds=list(min=log(0.00000001), max=log(0.1))),
    list(name="df_offset",     type="double", bounds=list(min=0.01, max=0.25))
  ),
  # Run the Optimization Loop between 10x - 20x the number of parameters
  observation_budget=60
))

# run experimentation loop
for(i in 1:experiment$observation_budget) {
  suggestion <- create_suggestion(experiment$id)

  opt_metric <- sentiment_metric(POSITIVE_TEXT, NEGATIVE_TEXT, suggestion$assignments)

  create_observation(experiment$id, list(
    suggestion=suggestion$id,
    value=opt_metric[1],
    value_stddev=opt_metric[2]
  ))
} # track progress on your experiment : https://sigopt.com/experiments
