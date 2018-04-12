[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# SigOpt Jupyter Notebook Example

Here we use SigOpt to optimze a simple 2D function within a [Jupyter notebook](http://jupyter.readthedocs.org/en/latest/install.html).

We create an experiment, form the suggestion feedback loop to optimize the function, then visualize the results against several other methods.

You can modify this notebook to optimize any function.

## Setup
1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `client_token` on the [API tokens page](https://sigopt.com/tokens) and set it
  as the `SIGOPT_API_TOKEN` environment variable.
3. Run `sudo ./setup_env.sh`

## Run
We recommend using [Jupyter](http://jupyter.readthedocs.org/en/latest/install.html) to walk through this example. Start Jupyter (run `jupyter notebook`), then open:

[`SigOpt_Introduction.ipynb`](https://github.com/sigopt/sigopt-examples/blob/master/jupyter-notebook-example/SigOpt_Introduction.ipynb)

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible.

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API and client libraries (Python, R, Java) integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

SigOpt is available through [Starter, Workgroup, and Enterprise plans](https://sigopt.com/pricing), and is [free forever for academic users](https://sigopt.com/edu).
