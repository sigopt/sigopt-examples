[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Building a Better Mousetrap with SigOpt

Example using SigOpt to solve a multicriteria optimization problem

More details about this example can be found in [the associated blog post](https://sigopt.com/blog/building-better-multicriteria-bayesian-optimization-algorith-sigopt/).

## Setup

1. Log in to your SigOpt account at [https://app.sigopt.com](https://app.sigopt.com)
2. Find your API Token on the [API tokens page](https://app.sigopt.com/tokens) and set it
  as the `SIGOPT_API_TOKEN` environment variable.
3. `git clone https://github.com/sigopt/sigopt-examples.git`
4. `cd sigopt-examples/maze-solving-mouse/`
5. Open a [Jupyter notebook](http://jupyter.org/) and start exploring.

## Optimize

If you execute any of the SigOpt experiments, you can track the progress on your [experiment dashboard](https://app.sigopt.com/experiments).

## Questions?
Visit the [SigOpt Community page](https://community.sigopt.com) and leave your questions.

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [API](https://docs.sigopt.com) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible.

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API and client libraries (Python, R, Java) integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more!
