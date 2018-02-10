[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# SigOpt IPython Notebook Example

Here we use SigOpt to optimze a simple 2D function within an [ipython notebook](http://ipython.org/notebook.html).

We create an experiment, form the suggestion feedback loop to optimize the function, then visualize the results against several other methods.

You can modify this notebook to optimize any function.

## Setup
1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `client_token` on your [user profile](https://sigopt.com/tokens/info) and set it
  as the `SIGOPT_API_TOKEN` environment variable.
4. Run `sudo ./setup_env.sh`

## Run
```
ipython notebook
```
This command will automatically open up your web browser. Navigate to SigOpt_Introduction.ipynb, and select Cell -> Run All from the menu bar.

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

SigOpt is available for a [30 day free trial](https://sigopt.com/signup), and is available [free forever for academic users](https://sigopt.com/edu).
