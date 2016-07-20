# SigOpt Random Forest Python Example

This example tunes a random forest using SigOpt + Python on the IRIS dataset. We offer:
 * Python API Client Example
 * Notebook Version of Python API Client Example
 * SigOpt + scikit-learn Integration Example

## Python
Add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) in line 16 of `random_forest.py`, then run the following code in a terminal to install dependencies and execute the script:

```
pip install sigopt
pip install sklearn
python random_forest.py
```

Learn more about our [Python API Client](https://sigopt.com/docs/overview/python).

## Notebook Version
We have a version of our Python Random Forest Example in a convenient [iPython notebook](https://ipython.org/) form.
To run, start up the notebook and add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) into the first cell. Run cells with `shift + enter`.

To run the notebook:

```
pip install jupyter
pip install sigopt
pip install sklearn
jupyter notebook
```

Learn more about our [Python API Client](https://sigopt.com/docs/overview/python) that we use in the notebook.


## Scikit-learn Integration
Add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) in line 15 of `random_forest.sklearn.py`, then run the following code in a terminal to install dependencies and execute the script:

```
pip install sigopt_sklearn
python random_forest.sklearn.py
```

Learn more about our [scikit-learn integration](https://github.com/sigopt/sigopt_sklearn).

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

SigOpt is available for a [30 day free trial](https://sigopt.com/signup), and is avaialable [free forever for academic users](https://sigopt.com/edu).
