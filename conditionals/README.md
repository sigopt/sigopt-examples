[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# SigOpt Conditional Experiments

Broadly speaking, a “conditional experiment” can turn “on” and “off” parameters subject to their satisfaction of a set of “conditions”.
Use cases include a deep learning model where, for example, `layer_3_nodes` is used by the model when `num_layers` is `3`, but is not used by the model when `num_layers` is `1` or `2`.

We have written examples using several different languages to demonstrate defining and running a SigOpt experiment with conditionals.

If you're interested in diving deeper, you can [read the docs](https://sigopt.com/docs/overview/conditionals). Or you can scroll down to find your language!

## [Python](python)
 * Python API Client

## [R](r)
 * R API Client

## [Java](java)
 * Java API Client

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
