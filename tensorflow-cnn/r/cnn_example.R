# Install packages
install.packages("devtools", repos = "http://cran.us.r-project.org")
library(devtools)
install.packages("RgoogleMaps", repos = "http://cran.us.r-project.org")
library(RgoogleMaps)
install.packages("R.matlab", repos = "http://cran.us.r-project.org")
library(R.matlab)
install_github("sigopt/SigOptR")
library(SigOptR)
install_github("rstudio/tensorflow")
library(tensorflow)

Sys.setenv(SIGOPT_API_TOKEN="INSERT_YOUR_TOKEN_HERE")

# load SVHN dataset
extra_X <- R.matlab::readMat('extra_32x32.mat')$X
extra_y <- R.matlab::readMat('extra_32x32.mat')['y']
test_X <- R.matlab::readMat('test_32x32.mat')$X
test_y <- R.matlab::readMat('test_32x32.mat')['y']
train_X <- R.matlab::readMat('train_32x32.mat')$X
train_y <- R.matlab::readMat('train_32x32.mat')['y']

convert_rgb2gray <- function(X) {
  X_gray <- array(0, c(32, 32, dim(X)[4]))
  for (i in 1:dim(X)[4]) {
    img_gray <- RGB2GRAY(X[,,,i])
    X_gray[,,i] <- img_gray
  }
  return(X_gray)
}

# convert to grayscale
extra_X <- convert_rgb2gray(extra_X)
test_X <- convert_rgb2gray(test_X)
train_X <- convert_rgb2gray(train_X)

image_w <- 32
extra_XZ <- array(aperm(extra_X), c(dim(extra_X)[3], image_w * image_w))
test_XZ <- array(aperm(test_X), c(dim(test_X)[3], image_w * image_w))
train_XZ <- array(aperm(train_X), c(dim(train_X)[3], image_w * image_w))

# normalize image features
extra_XZ <- t(scale(t(extra_XZ)))
test_XZ <- t(scale(t(test_XZ)))
train_XZ <- t(scale(t(train_XZ)))

# one-hot encode labels
extra_y$y <- factor(extra_y$y)
extra_yZ <- model.matrix(~.-1, data = extra_y)
test_y$y <- factor(test_y$y)
test_yZ <- model.matrix(~.-1, data = test_y)
train_y$y <- factor(train_y$y)
train_yZ <- model.matrix(~.-1, data = train_y)

# stack train, extra
extra_XZ <- rbind(extra_XZ, train_XZ)
extra_yZ <- rbind(extra_yZ, train_yZ)

# take 75% of dataset
index <- sample(1:nrow(extra_XZ), size=0.75 * nrow(extra_XZ))
extra_XZ <- extra_XZ[index, ]
index <- sample(1:nrow(extra_yZ), size=0.75 * nrow(extra_yZ))
extra_yZ <- extra_yZ[index, ]

# create SigOpt experiment
experiment <- create_experiment(list(
  name="SVHN ConvNet (R)",
  parameters=list(
    list(name="filter1_w", type="int", bounds=list(min=3, max=10)),
    list(name="filter1_depth", type="int", bounds=list(min=10, max=64)),
    list(name="filter2_w", type="int", bounds=list(min=3, max=10)),
    list(name="filter2_depth", type="int", bounds=list(min=10, max=64)),
    list(name="drp_out_keep_p", type="double", bounds=list(min=0.2, max=1.0)),
    list(name="log_rms_lr", type="double", bounds=list(min=log(0.0001), max=log(1.0))),
    list(name="rms_mom", type="double", bounds=list(min=0.5, max=1.0)),
    list(name="rms_decay", type="double", bounds=list(min=0.5, max=1.0))
  ),
  observation_budget=100
))

evaluate_metric <- function(extra_XZ, extra_yZ, params, image_w) {
  sess <- tf$InteractiveSession()
  x <- tf$placeholder(tf$float32, shape(NULL, image_w * image_w))
  y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))
  filter1_w <- as.integer(params$filter1_w)
  filter1_depth <- as.integer(params$filter1_depth)
  filter2_w <- as.integer(params$filter2_w)
  filter2_depth <- as.integer(params$filter2_depth)
  drp_out_keep_p <- params$drp_out_keep_p
  rms_lr <- exp(params$log_rms_lr)
  rms_mom <- params$rms_mom
  rms_decay <- params$rms_decay

  weight_variable <- function(shape) {
    initial <- tf$truncated_normal(shape, stddev=0.1)
    return(tf$Variable(initial))
  }

  bias_variable <- function(shape) {
    initial <- tf$constant(0.1, shape=shape)
    return(tf$Variable(initial))
  }

  conv2d <- function(x, W) {
    return(tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME'))
  }

  max_pool_2x2 <- function(x) {
    return(tf$nn$max_pool(x, ksize=c(1L, 2L, 2L, 1L),
                          strides=c(1L, 2L, 2L, 1L), padding='SAME'))
  }

  W_conv1 <- weight_variable(shape(filter1_w, filter1_w, 1L, filter1_depth))
  b_conv1 <- bias_variable(shape(filter1_depth))

  x_image <- tf$reshape(x, shape(-1L, image_w, image_w, 1L), name='reshape1')

  h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 <- max_pool_2x2(h_conv1)

  W_conv2 <- weight_variable(shape(filter2_w, filter2_w, filter1_depth, filter2_depth))
  b_conv2 <- bias_variable(shape(filter2_depth))

  h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 <- max_pool_2x2(h_conv2)

  W_fc1 <- weight_variable(shape(8L * 8L * filter2_depth, 1024L))
  b_fc1 <- bias_variable(shape(1024L))

  h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, 8L * 8L * filter2_depth), name='reshape2')
  h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob <- tf$placeholder(tf$float32)
  h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)

  W_fc2 <- weight_variable(shape(1024L, 10L))
  b_fc2 <- bias_variable(shape(10L))

  y_conv <- tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)

  cross_entropy <- -tf$reduce_sum(y_ * tf$log(y_conv))
  train_step <- tf$train$RMSPropOptimizer(rms_lr, decay=rms_decay, momentum=rms_mom)$minimize(cross_entropy)

  correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
  accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

  # create cross-validation fold
  index <- sample(1:nrow(extra_XZ), size=0.15 * nrow(extra_XZ))
  train_XZ_2 <- extra_XZ[-index, ]
  valid_XZ <- extra_XZ[index, ]
  index <- sample(1:nrow(extra_yZ), size=0.15 * nrow(extra_yZ))
  train_yZ_2 <- extra_yZ[-index, ]
  valid_yZ <- extra_yZ[index, ]

  sess$run(tf$global_variables_initializer())

  # stochastic gradient descent
  t0 <- Sys.time()
  batch_size <- 10000
  epoch_size <- 1
  for (k in 1:epoch_size) {
    for (i in seq(1, dim(train_XZ_2)[1] - batch_size, by=batch_size)) {
      if (i/batch_size %% 100 == 0) {
        train_accuracy <- accuracy$eval(feed_dict = dict(
          x = train_XZ_2[i:(i + batch_size), ], y_ = train_yZ_2[i:(i + batch_size), ], keep_prob = 1.0
        ))
        print(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
      }
      train_step$run(feed_dict = dict(
        x = train_XZ_2[i:(i + batch_size), ], y_ = train_yZ_2[i:(i + batch_size), ], keep_prob = drp_out_keep_p
      ))
    }
  }

  # chunk to not run out of memory
  opt_metric <- 0.0
  opt_chunk <- 10L
  step_size <- dim(valid_XZ)[1] / opt_chunk
  for (i in seq(1, dim(valid_XZ)[1] - step_size, by=step_size)) {
    chunk_acc <- accuracy$eval(feed_dict = dict(
      x = valid_XZ[i:(i + step_size), ], y_ = valid_yZ[i:(i + step_size), ], keep_prob = 1.0
    ))
    chunk_range <- min(i + step_size, dim(valid_XZ)[1]) - i
    chunk_perc <- chunk_range / as.double(dim(valid_XZ)[1])
    opt_metric <- opt_metric + (chunk_acc * chunk_perc)
  }
  print(opt_metric)
  print(sprintf("Total Time: %s\n", difftime(Sys.time(), t0)))
  sess$close()
  return(opt_metric)
}


# SigOpt optimization loop
for (i in 1:experiment$observation_budget) {
  suggestion <- create_suggestion(experiment$id)
  params <- suggestion$assignments

  opt_metric <- evaluate_model(extra_XZ, extra_yz, params, image_w)

  create_observation(experiment$id, list(
    suggestion=suggestion$id,
    value=as.double(opt_metric),
    value_stddev=0.05
  ))
}
