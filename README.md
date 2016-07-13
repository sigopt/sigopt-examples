# Getting Started with SigOpt

Welcome to SigOpt Examples. These examples show you how to use [SigOpt](https://sigopt.com) for model tuning tasks in various machine learning environments. 

## Requirements

These examples will run on any Linux or Mac OS X machine from the command line. Each example contains a README.md with specific setup instructions.

## First Time?

If this is your first time using SigOpt, we recommend you work through the [Random Forest](random-forest) example. In this example, you will use a random forest to classify data from the [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) and use SigOpt maximize the k-fold cross-validation accuracy by tuning the model's hyperparameters. This example is available in a wide variety of languages and integrations:
 * Python
 * R
 * scikit-learn integration
 * iPython notebook

If this is your first time using SigOpt, we recommend you work through the [Python Text Classifier](text-classifier) example. In this example you will create a logistic regression model to classify Amazon product reviews and use SigOpt maximize the k-fold cross-validation accuracy by tuning the regression coefficients and feature parameters.

## More Examples

- [ipython-notebook-example](https://github.com/sigopt/sigopt-examples/tree/master/ipython-notebook-example): Simple example of using SigOpt to optimize a 2D function with plots and comparisons in an iPython Notebook.
- [java](https://github.com/sigopt/sigopt-examples/tree/master/java): An example of using the Java API client.
- [sigopt-beats-vegas](https://github.com/sigopt/sigopt-examples/tree/master/sigopt-beats-vegas): Using SigOpt to tune a model to beat the Vegas odds in Python ([blog post](http://blog.sigopt.com/post/136340340198/sigopt-for-ml-using-model-tuning-to-beat-vegas)).
- [text-classifier](https://github.com/sigopt/sigopt-examples/tree/master/text-classifier): Example using SigOpt to tune a text classifier in Python ([blog post](http://blog.sigopt.com/post/133089144983/sigopt-for-ml-automatically-tuning-text)).
- [unsupervised-model](https://github.com/sigopt/sigopt-examples/tree/master/unsupervised-model): Example using SigOpt and xgboost to tune a combined unsupervised and supervised model for optical character recognition ([blog post](http://blog.sigopt.com/post/140871698423/sigopt-for-ml-unsupervised-learning-with-even))
- [tensorflow-cnn](https://github.com/sigopt/sigopt-examples/tree/master/tensorflow-cnn): Example using SigOpt and TensorFlow to tune a convolutional neural network's structure and gradient descent algorithm ([blog post](http://blog.sigopt.com/post/141501625253/sigopt-for-ml-tensorflow-convnets-on-a-budget))
- [classifier](https://github.com/sigopt/sigopt-examples/tree/master/classifier): Using SigOpt to tune a machine learning classifier in Python ([blog post](http://blog.sigopt.com/post/111903668663/tuning-machine-learning-models)).
- [parallel](https://github.com/sigopt/sigopt-examples/tree/master/parallel): Examples of running SigOpt from multiple parallel processes in Python ([blog post](http://blog.sigopt.com/post/111903668663/tuning-machine-learning-models)).
- [other-languages](https://github.com/sigopt/sigopt-examples/tree/master/other-languages): Example of using the python client to run an evaluation function in a different language.
- [random-forest](https://github.com/sigopt/sigopt-examples/tree/master/random-forest): Example of tuning the hyperparameters of a random forest on the open IRIS dataset in a variety of languages and integrations including R and Python.

If you have any questions, comments, or concerns please email us at [contact@sigopt.com](mailto:contact@sigopt.com)
