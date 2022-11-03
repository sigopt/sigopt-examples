[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Text Classifier Tuning Python Example

Example using SigOpt and Python to tune logistic regression model for text sentiment classification.

More details about this example can be found in [the associated blog post](https://sigopt.com/blog/automatically-tuning-text-classifiers/).

## Setup

1. Log in to your SigOpt account at [https://app.sigopt.com](https://app.sigopt.com)
2. Find your API Token on the [API tokens page](https://app.sigopt.com/tokens) and set it
  as the `SIGOPT_API_TOKEN` environment variable.
3. `git clone https://github.com/sigopt/sigopt-examples.git`
4. `cd sigopt-examples/text-classifier/python`
5. `sudo ./setup_env.sh`

## Run

We recommend using [Jupyter](http://jupyter.readthedocs.org/en/latest/install.html) to walk through this example. Start Jupyter (run `jupyter notebook`), then open:

[`SigOpt Text Classifier Walkthrough.ipynb`](https://github.com/sigopt/sigopt-examples/blob/master/text-classifier/SigOpt%20Text%20Classifier%20Walkthrough.ipynb)

The classifier tuning can also be run without Jupyter with this command:

```
nohup python sentiment_classifier.py &
```

## Optimize

Once the text classifier model tuning loop is running, you can track the progress on your [experiment dashboard](https://sigopt.com/experiments).

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
