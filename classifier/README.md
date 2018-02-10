[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Classifier Tuning

Machine learning classifier hyperparameter optimization example.

More details about this example can be found in [the associated blog post](http://blog.sigopt.com/post/111903668663/tuning-machine-learning-models).

## Setup
1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `API_TOKEN` on the [API tokens page](https://sigopt.com/tokens/info).
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
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

SigOpt is available for a [30 day free trial](https://sigopt.com/signup), and is available [free forever for academic users](https://sigopt.com/edu).
