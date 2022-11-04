[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Using SigOpt's XGBoost Integration for Fraud Classification

This tutorial uses SigOpt's [XGBoost integration](https://docs.sigopt.com/ai-module-api-references/xgboost) to optimize a fraud classifier. You will need a [SigOpt account](https://sigopt.com/signup) and SigOpt [API token](https://app.sigopt.com/tokens/info), as well as a Kaggle account and Kaggle API token in order to run this code.

## Jupyter Setup

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/xgboost-integration-examples`
3. `pip install -r requirements.txt` if you do not have all required packages installed
4. Run `jupyter notebook` in that directory and open xgb-integration-py-class.ipynb or xgb-integration-py-reg.ipynb in the web interface
5. Run all cells or step through the notebook


## Optimize

Check the progress of your SigOpt runs and experiments on your [experiment dashboard](https://sigopt.com/experiments).

## Questions?
Visit the [SigOpt Community page](https://community.sigopt.com) and leave your questions.

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [API](https://docs.sigopt.com) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more and sign up today!
