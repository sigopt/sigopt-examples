[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Metric Constraints Demo with Tensorflow/Keras MicronNet

This example showcases the [Metric Constraints](https://docs.sigopt.com/advanced_experimentation/metric_constraints) feature in SigOpt, as described in this [blog post](https://sigopt.com/blog/metric-constraints-demo/). We use the Metric Constraints feature to optimize for the top-1 accuracy of a CNN with a constraint of the size of the network. We demonstrate this feature using the German Traffic Signs Dataset (GTSRB).

The CNN model is inspired by the [*MicronNet*](https://arxiv.org/abs/1804.00497) model.
## Run in Google Colab

To run the notebooks in Google Colab, click the `Open in Colab` button in each notebook.

* The `GTSRB_preprocessing_augmentation.ipynb` notebook preprocesses the dataset.
* The `sigopt_metric_constraints_demo.ipynb` notebook runs the SigOpt experiment of tuning the neural network.

## Run locally

Alternatively, you can also run these notebooks locally in Jupyter notebooks. You will need to install the following dependencies. Note that this example uses the Tensorflow2/Keras API.

```
matplotlib
numpy
scikit-image
sigopt
tensorflow
```

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
