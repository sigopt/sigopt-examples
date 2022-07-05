[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Using SigOpt's XGBoost Integration for Regression

This tutorial uses SigOpt's XGBoost integration to optimize a XGBoost regressor model. You will need a [SigOpt account](https://sigopt.com/signup) and SigOpt [API token](https://app.sigopt.com/tokens/info), as well as a Kaggle account and Kaggle API token in order to run this code.

## Jupyter Setup

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/xgb-integration-py-reg`
3. `pip install -r requirements.txt` if you do not have all required packages installed
4. Run `jupyter notebook` in that directory and open xgb-integration-py-reg.ipynb in the web interface
5. Run all cells or step through the notebook

## Python Setup

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/xgb-integration-py-reg`
3. `pip install -r requirements.txt` if you do not have all required packages installed
4.  Uncomment and add your SigOpt [API token](https://app.sigopt.com/tokens/info) where you see `YOUR_API_TOKEN_HERE`
5. Run `python xgb-integration-py-reg.py` or open the file in your favorite text editor to see what it does

## Optimize

Check the progress of your SigOpt runs and experiments on your [experiment dashboard](https://sigopt.com/experiments).

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more and sign up today!
