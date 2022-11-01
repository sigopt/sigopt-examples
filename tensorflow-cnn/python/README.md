[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# TensorFlow Convolutional Neural Network Tuning Python Example

Example using SigOpt to tune a convolutional neural network for OCR recognition

More details about this example can be found in [the associated blog post](http://blog.sigopt.com/post/141501625253/sigopt-for-ml-tensorflow-convnets-on-a-budget).

## Setup

1. Log in to your SigOpt account at [https://app.sigopt.com](https://app.sigopt.com)
2. Find your API Token on the [API tokens page](https://app.sigopt.com/tokens).
3. `git clone https://github.com/sigopt/sigopt-examples.git`
4. `cd sigopt-examples/tensorflow-cnn/`
5. `sudo ./setup.sh`

## Optimize

Once the CNN model is being optimized, you can track the progress on your [experiment dashboard](https://sigopt.com/experiments).

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [API](https://docs.sigopt.com) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible.

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more!
