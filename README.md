[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Getting Started with SigOpt

Welcome to the SigOpt Examples. These examples show you how to use [SigOpt](https://sigopt.com) for model tuning tasks in various machine learning environments.

## Requirements

Most of these examples will run on any Linux or Mac OS X machine from the command line. Each example contains a README.md with specific setup instructions.

## First Time?

If this is your first time using SigOpt, we recommend you work through the [Random Forest](random-forest) example. In this example, you will use a random forest to classify data from the [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) and use SigOpt to maximize the k-fold cross-validation accuracy by tuning the model's hyperparameters. This example is available in a wide variety of languages and integrations:
 * [Python](random-forest/python)
 * [R](random-forest/r)
 * [Java](random-forest/java)
 * [Jupyter notebook](random-forest/python#notebook-version)
 * [scikit-learn integration](random-forest/python#scikit-learn-integration)

## More Examples

- [sigopt-beats-vegas](sigopt-beats-vegas): Using SigOpt to tune a model to beat the Vegas odds in Python ([blog post](http://blog.sigopt.com/post/136340340198/sigopt-for-ml-using-model-tuning-to-beat-vegas)).
- [text-classifier](text-classifier): Example using SigOpt to tune a text classifier in Python and R ([blog post](http://blog.sigopt.com/post/133089144983/sigopt-for-ml-automatically-tuning-text)).
- [unsupervised-model](unsupervised-model): Example using SigOpt and xgboost to tune a combined unsupervised and supervised model for optical character recognition ([blog post](http://blog.sigopt.com/post/140871698423/sigopt-for-ml-unsupervised-learning-with-even))
- [tensorflow-cnn](tensorflow-cnn): Example using SigOpt and TensorFlow to tune a convolutional neural network's structure and gradient descent algorithm ([blog post](http://blog.sigopt.com/post/141501625253/sigopt-for-ml-tensorflow-convnets-on-a-budget))
- [classifier](classifier): Using SigOpt to tune a machine learning classifier in Python ([blog post](http://blog.sigopt.com/post/111903668663/tuning-machine-learning-models)).
- [parallel](parallel): Examples of running SigOpt from multiple parallel processes in Python ([blog post](http://blog.sigopt.com/post/111903668663/tuning-machine-learning-models)).

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible.

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API and client libraries (Python, R, Java) integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more!
