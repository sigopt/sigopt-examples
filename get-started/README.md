[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Getting Started with SigOpt

These notebook examples use SigOpt to track and optimize an XGBoost classifier model on the sklearn wine dataset in Python.

## Google Colab

1. Open the SigOpt Runs demo in https://colab.research.google.com/github/sigopt/sigopt-examples/blob/master/get-started/sigopt_runs_demo.ipynb
2. Open the SigOpt Experiment and Optimization demo in https://colab.research.google.com/github/sigopt/sigopt-examples/blob/master/get-started/sigopt_experiment_and_optimization_demo.ipynb

## Jupyter Setup

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/get-started`
3. Run `jupyter lab` in that directory and open sigopt_runs_demo.ipynb or sigopt_experiment_and_optimization_demo.ipynb in the web interface
4. Add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) when prompted after running the `!sigopt config` command
5. Run all cells or step through the notebook

## Questions?
Visit the [SigOpt Community page](https://community.sigopt.com) and leave your questions.

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more!
