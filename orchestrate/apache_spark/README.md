[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)
# Apache Spark with Sigopt Orchestrate

SigOpt Orchestrate is a command-line tool for managing training clusters and running optimization experiments.

In this example, you'll spin up an Orchestrate cluster with `i3` nodes,
then launch a hyperparameter tuning job for a model trained with Apache Spark.

## Get Started

This example asssumes that you have already
[installed SigOpt Orchestrate](https://app.sigopt.com/docs/orchestrate/installation).

Follow the steps below to run this example:

```bash
git clone https://github.com/sigopt/sigopt-examples.git  # Clone this repo
cd sigopt-examples/orchestrate
sigopt cluster create -f clusters/i3_cluster.yml  # Create the cluster
sigopt run --directory apache_spark -f apache_spark/orchestrate.yml  # Run the experiment
```

To monitor status and view logs, use the experiment id that is outputted by the `sigopt run` command:

```bash
sigopt logs $experiment_id
sigopt status $experiment_id
```

And, don't forget to destroy your cluster when you're all done:

```bash
sigopt cluster destroy -n i3-cluster
```

## Not Using Spark 2.4.0?

This example uses Apache Spark 2.4.0.
To use a different version, simply edit the experiment configuration file, [orchestrate.yml](orchestrate.yml),
and re-run your experiment.

## Join the Orchestrate Alpha

SigOpt Orchestrate is now in alpha release, and we're looking for partners to use our tool and give us feedback!

If you're interested in joining our alpha testing of SigOpt Orchestrate, email us at <support@sigopt.com> to set up a demo.

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API and client libraries (Python, R, Java) integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

SigOpt is available through [Starter, Workgroup, and Enterprise plans](https://sigopt.com/pricing), and is [free forever for academic users](https://sigopt.com/edu).
