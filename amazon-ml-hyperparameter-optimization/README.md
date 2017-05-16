# Performing Hyperparameter Optimization with Amazon Machine Learning

This sample code builds a hyperparameter optimization pipeline for Amazon Machine Learning using the latest AWS SDK for Python (Boto 3). The user can optionally specify hyperparameters upfront for manual tuning or use [SigOpt](https://www.sigopt.com)'s API for Bayesian optimization.

This example is directly based off of Amazon's [K-Fold Cross Validation](https://github.com/awslabs/machine-learning-samples/tree/master/k-fold-cross-validation) example.

## Setting Up

### Install dependencies

This sample script was developed and tested in python 2.

This sample script depends on the `boto3` and `sigopt` packages. If you have [pip](https://pip.pypa.io/en/stable/) installed you can install dependencies by running

```
pip install -r requirements.txt
```

### Configure AWS Credentials

Your AWS credentials must be stored in a `~/.boto` or `~/.aws/credentials` file. Your credential file should look like this:

```
  [Credentials]
  aws_access_key_id = YOURACCESSKEY
  aws_secret_access_key = YOURSECRETKEY
```

To learn more about configuring your AWS credentials with Boto 3, go to [Boto 3 Quickstart](http://boto3.readthedocs.io/en/latest/guide/quickstart.html).

### Configure SigOpt Credentials (optional)

If want to use SigOpt to optimize your hyperparameters faster and better than tuning by hand, sign up for a free trial on our [website]([SigOpt](https://www.sigopt.com)) and grab your API token from your [user profile](https://www.sigopt.com/user/profile).

### Get the Code

Get the samples by cloning this repository.

```
  git clone https://github.com/alexandraj777/machine-learning-samples.git
```

## Demo

The basic demo is in the script `hyperparameter_optimization.py`. This script relies on a manually specified list of hyperparameters to tune the hyperparameters of a linear binary classification model. Edit the hyperparameter assignments in `hyperparameters.py` to manually tune your model.

The arguments to `hyperparameter_optimization.py` are the number of folds (required), and an optional resource prefix.

```
python hyperparameter_optimization.py --name 4-fold-hy-opt-demo 4
```

The script `hyperparameter_optimization_with_sigopt.py` uses the [SigOpt](https://www.sigopt.com) API to optimally suggest hyperparameters for the model.


Run the script `hyperparameter_optimization_with_sigopt.py` with all of the options in the above section, and additionally provide your SigOpt API token.

```
python hyperparameter_optimization_with_sigopt.py --name 4-fold-hy-opt-sigopt-demo 4 --sigopt-api-token <SIGOPT_API_TOKEN>
```

### How it Works

Many machine learning models have exposed parameters, commonly known as hyperparameters (Amazon ML sometimes calls them training parameters), that you choose values for before model training begins. Finding the best values of these hyperparameters is important because they have  large impact on the performance of the model. The search for the best assignments of hyperparameters for a model is commonly known as hyperparameter optimization.

The [hyperparameters](http://docs.aws.amazon.com/machine-learning/latest/dg/training-parameters.html) exposed by Amazon ML are limited, so we only tune the regularization type and regularization amount. You'll notice that learning rate is also a hyperparameter of your model, but Amazon ML is automatically selecting a value for it based on your data, so we can't tune it in this example.

Regularization roughly quantifies the idea in machine learning that simpler explanations are better. This [Quora answer](https://www.quora.com/What-is-regularization-in-machine-learning) has a great explanation of what regularization is and why it matters.

Hyperparameter optimization works to maximize some performance metric of your model. This example maximizes the Area Under the (Receiver Operating Characteristic) Curve (AUC) of a binary classifier because it is easily exposed by Amazon Machine Learning. Alternatives include maximizing accuracy or minimizing error (maximizing negative error). Read more about how Amazon measures model performance on [Binary Model Insights](http://docs.aws.amazon.com/machine-learning/latest/dg/binary-model-insights.html).

In keeping with best practices for hyperparameter optimization, and to prevent over-fitting of the model, this example actually maximizes the average of k-fold cross validated AUC metrics. This description skips over the details of how cross validation is performed with Amazon ML, because it is described much better in the README for the [K-Fold Cross Validation](https://github.com/awslabs/machine-learning-samples/tree/master/k-fold-cross-validation) example that formed the basis for this code sample.

To perform the hyperparameter optimization the scripts iteratively choose new values of `regularization_type` and `regularization_amount`, evaluate a model with these new hyperparameters for every fold of the data, average the AUC metrics, and record the performance of the assignments. At the end of some number of evaluations, the assignments that produced the best model performance are selected. The final model will use these hyperparameters and be trained on all of the available data.

If you optimize with SigOpt you can see the best hyperparameters on your [dashboard](https://www.sigopt.com/experiments).

### Under the Hood

Assume `k` is the number of folds for cross validation. At startup each script will create `2k` datasources on Amazon ML: a train datasource and a complementary evaluation datasource for each fold. The datasources can be reused throughout hyperparameter optimization and are cleaned up at the end of the script.

Next, the script grabs the next assignments for hyperparameters `regularization_type` and `regularization_amount`. In the basic script these hyperparameter are imported from `hyperparameters.py`; in the SigOpt example they are created via the SigOpt API. Each time the script evaluates a model on a new set of hyperparameters is creates `k` machine learning models, one for each train datasource, and `k` evaluations, one for each evaluate datasource, on Amazon ML. These objects cannot be reused and are deleted once all evaluations have completed.

After evaluations are created the script spawn threads to poll the Amazon ML API until the evaluations have status `"COMPLETED"`. Once every thread has completed we compute the average and standard deviation of the `k` AUC metrics. At this point the script is ready to grab the next assignments of hyperparameters and repeat. At SigOpt, we call this process the **optimization loop**.

### Parallelism

Amazon Machine Learning asynchronously creates datasources, machine learning models, and evaluations. API calls via the python SDK will return quickly so that you can build a datasource, machine learning model, and an evaluation while the datasource is still pending! Since your machine is not doing the heavy computation of training and testing the model, it has great opportunities for parallelization, splitting up the optimization loop between `n` different threads or processes.

If you're running the SigOpt example, read how easy it is to [parallelize hyperparameter optimization with SigOpt](https://sigopt.com/docs/overview/parallel).

### Note from Amazon

All resources created by these scripts are billed at the regular Amazon ML rates. For information about Amazon ML pricing, see [Amazon Machine Learning Pricing](https://aws.amazon.com/machine-learning/pricing/). For information on how to delete your resources, see [Clean Up](http://docs.aws.amazon.com/machine-learning/latest/dg/step-6-clean-up.html) in the tutorial in the *Amazon Machine Learning Developer Guide*.
