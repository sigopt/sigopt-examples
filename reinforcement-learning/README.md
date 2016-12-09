[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Tuning Deep Q-Networks With SigOpt

This example uses SigOpt to tune a Deep Q-Network (DQN) to solve a reinforcement learning problem using OpenAI's gym simulation environments.

More details about this example can be found in [the associated blog post](http://blog.sigopt.com/post/154251615358/sigopt-for-ml-using-bayesian-optimization-for-reinforcement-learning).

## Setup

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/reinforcement-learning`
3. Install requirements. For Linux: `sudo ./setup-linux.sh`. For Mac OS X: `sudo ./setup-osx.sh`
4. Add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) to line 9 of `dqn.py`
5. Execute the script by running: `python dqn.py`
6. Once the SigOpt optimization loop is initiated, you can track the progress on your [experiment dashboard](https://sigopt.com/experiments).

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
