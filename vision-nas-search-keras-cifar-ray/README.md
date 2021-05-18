[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Neural Architecture Search to Train a Computer Vision Classifier in Keras on CIFAR10 Data using Ray Tune and SigOpt

This example uses SigOpt to tune Neural Architecture search, using a Keras image classifier. Ray Tune is used to orchestrate training, and uses its built in interface to the SigOpt API.

## Jupyter Setup

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/vision-nas-search-keras-cifar-ray`
3. Run `jupyter lab` in that directory and open xgboost_py_classifier.ipynb in the web interface
4. Add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) in the Jupyter cell where you see `YOUR_API_TOKEN_HERE`
5. Run all cells or step through the notebook

## Optimize

Once the SigOpt optimization loop is initiated, you can track the progress on your [experiment dashboard](https://sigopt.com/experiments).

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more, and be sure to [sign up for a free account](https://app.sigopt.com/signup) if you haven't already.
