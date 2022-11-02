[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Classifier Tuning

Machine learning classifier hyperparameter optimization example.

More details about this example can be found in [the associated blog post](https://sigopt.com/blog/tuning-machine-learning-models/).

## Setup
1. Log in to your SigOpt account at [https://app.sigopt.com](https://app.sigopt.com)
2. Find your API Token on the [API tokens page](https://app.sigopt.com/tokens).
3. Install requirements `pip install -r requirements.txt`

## Run

Run default example using small sklearn dataset and Gradient Boosting Classifier.

```bash
python classifier_tuner.py --client-token API_TOKEN
```

Run using connect-4 dataset (this takes a long time) and Support Vector Classfier

```bash
python classifier_tuner.py --classifier-type SVC --dataset-name connect-4 --test-set-size 7557 --client-token API_TOKEN
```

## Questions?
Visit the [SigOpt Community page](https://community.sigopt.com) and leave your questions.

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [API](https://docs.sigopt.com) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible.

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API and client libraries (Python, R, Java) integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more!
