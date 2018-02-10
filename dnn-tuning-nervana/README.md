[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Using SigOpt and Nervana Cloud to tune DNNs

Learn more at the associated blog post: [Much Deeper, Much Faster: Deep Neural Network Optimization with SigOpt and Nervana Cloud](http://blog.sigopt.com/post/146208659358/much-deeper-much-faster-deep-neural-network).

## SigOpt Setup

1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `client_token` on the [API tokens page](https://sigopt.com/tokens) and set it
  as the `SIGOPT_API_TOKEN` environment variable.

## Ncloud Setup

First, contact products@nervanasys.com for login credentials. Then install Ncloud

```
pip install ncloud
```

Finally, enter your login credentials with the following
```
ncloud configure
```

## Running on Nervana Cloud

```
pip install sigopt
python sigopt_nervana.py
```

## Running on AWS with neon

Launch the AWS AMI with ID `ami-d7562bb7` named `Nervana neon and ncloud`
   available in N. California region. g2.2xlarge instance type is recommended.

```
source ./neon/.venv/bin/activate
pip install sigopt
./neon/neon/data/batch_writer.py --set_type cifar10 --data_dir "/home/ubuntu/data" --macro_size 10000 --target_size 40
python nervana_all_cnn.py
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
