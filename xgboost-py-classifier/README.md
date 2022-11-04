[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# XGBoost Supervised Python Jupyter Notebook Classifier Tuning

This example uses SigOpt to tune an XGBoost classifier on the Wisconsin breast cancer dataset in Python.

## Jupyter Setup

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/xgboost-py-classifier`
3. Run `jupyter lab` in that directory and open xgboost_py_classifier.ipynb in the web interface
4. Add your SigOpt [API token](https://docs.sigopt.com/core-module-api-references/get_started) in the Jupyter cell where you see `YOUR_API_TOKEN_HERE`
5. Run all cells or step through the notebook

## Optimize

Once the SigOpt optimization loop is initiated, you can track the progress on your [experiment dashboard](https://sigopt.com/experiments).

## Questions?
Visit the [SigOpt Community page](https://community.sigopt.com) and leave your questions.

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [API](https://docs.sigopt.com) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more!
