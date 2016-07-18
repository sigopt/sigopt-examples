# Using SigOpt and Nervana Cloud to tune DNNs

Learn more at the associated blog post: [Much Deeper, Much Faster: Deep Neural Network Optimization with SigOpt and Nervana Cloud](http://blog.sigopt.com/post/146208659358/much-deeper-much-faster-deep-neural-network).

## SigOpt Setup

1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile) and set it
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
