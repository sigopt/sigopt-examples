
# Classifying the Stanford Cars Dataset

This repository runs hyperparameter optimization on tuning pretrained models from the [PyTorch model zoo](https://pytorch.org/docs/stable/torchvision/models.html) to classify images of cars in the Stanford Cars dataset.
This repository offers the option to tune only the fully connected layer of the pretrained network or fine tune the whole network.
The pretrained models supported are Resnet 18 and ResNet 50 are trained on ImageNet-1000.


## Getting Started

### Clone repository

```buildoutcfg

git clone https://github.com/sigopt/sigopt-examples.git

mkdir -p sigopt-examples/stanford-car-classification/data

cd sigopt-examples

```

### Download data

The Stanford Cars dataset can be found [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

The dataset includes:
* images of cars (cars_ims.tgz)
* labels (cars_annos.mat)
* devkit including human readable labels (cars_meta.mat)

```buildoutcfg

wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz -P stanford-car-classification/data

wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat -P stanford-car-classification/data

wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz -P stanford-car-classification/data

```

or using CURL:

```buildoutcfg
curl http://imagenet.stanford.edu/internal/car196/car_ims.tgz -o stanford-car-classification/data/car_ims.tgz

curl http://imagenet.stanford.edu/internal/car196/cars_annos.mat -o stanford-car-classification/data/cars_annos.mat

curl https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz -o stanford-car-classification/data/car_devkit.tgz

```

Unzip folders:

```buildoutcfg
tar -C stanford-car-classification/data -xzvf stanford-car-classification/data/car_ims.tgz
tar -C stanford-car-classification/data -xzvf stanford-car-classification/data/car_devkit.tgz

Only keeping the meta file in the devkit:

mv ./stanford-car-classification/data/devkit/cars_meta.mat ./stanford-car-classification/data
rm -r ./stanford-car-classification/data/devkit

```

## Virtual environment set up

### Creating a new virtualenv

#### Installing pip3

MacOS:

```
brew install python3
```

Ubuntu:

```
sudo apt-get update
sudo apt-get -y install python3-pip

```

#### Installing virtualenv

```buildoutcfg
pip3 install virtualenv

```

#### Set up new virtualenv

```buildoutcfg
python3 -m virtualenv [PATH TO VIRUALENV]

example:

python3 -m virtualenv ./stanford-car-classification-venv

```
#### Installing requirements in virtualenvironment

 
```buildoutcfg
source [PATH TO VIRTUALENV]/bin/activate (ex: ./stanford-car-classification-venv/bin/activate)

pip install orchestrate
pip install sklearn
pip install matplotlib
pip install torch torchvision
pip install botocore

```

## Tuning Pre-trained ResNet Models

### CommandLine Interface

```
python run_resnet_training_cli.py --path_images <path to parent directory of car_ims folder>
--path_data <path to cars_meta.mat> 
--path_labels <path to cars_annos.meta>
[--path_model_checkpoint <path to model checkpointing directory, default: No checkpointing>] 
[--checkpoint_frequency <frequency to generate PyTorch checkpoint files>, default: No checkpointing]
--model {ResNet18 | ResNet50}
--epochs <number of epochs to train model>
--validation_frequency <frequency to run validation during training> 
--number_of_classes <number of labels in data set>
--data_subset <subset of data to be used [0, 1.0]> 
--learning_rate_scheduler <learning rate annealing factor (new learning rate = leanring rate * learning rate scheduler)>
--batch_size <batch size (will be applied as a factor of 2, ex: batch size = 2, 2^2 =4)>
--weight_decay <weight decay (log base e value expected)>
--momentum <momentum value>
--learning_rate <learning rate value (log base e value expected)>
--scheduler_rate <patience used to apply learning rate scheduler>
{--nesterov | --no-nesterov} {--freeze_weights | --no-freeze_weights}

```

To include Nesterov in the learning: --nesterov must be included

To not include Nesterov: --no-nesterov must be included

To train the fully connected layer: --freeze_weights must be included

To fine tune the whole network: --no-freeze_weights must be included

The data_subset option is used to specify the fraction of the Stanford Cars dataset to use (data_subset = 0.5 means 50% of the data is used).
A 20% validation split is applied to the data used.

Logs will be outputted to the working directory.

An output directory with the format `<time since epoch in seconds>_model` will be created under the current working directory.
Checkpoints will be stored under `<time since epoch in seconds>_model/model_checkpoints`.

#### Tuning the Fully Connected Layer Only

Example: 

```
source ./stanford-car-classification-venv/bin/activate
python run_resnet_training_cli.py --path_images ./stanford-car-classification/data/ --path_data ./stanford-car-classification/data/cars_annos.mat --path_labels ./stanford-car-classification/data/cars_meta.mat --path_model_checkpoint ./stanford-car-classification --checkpoint_frequency 10 --model ResNet18 --epochs 35 --validation_frequency 10  --number_of_classes 196 --data_subset 1.0 --learning_rate_scheduler 0.2 --batch_size 6 --weight_decay 0.80 --momentum 0.9 --learning_rate 0.04 --scheduler_rate 5 --nesterov --freeze_weights

```

The above example tunes the fully connected layer of a pretrained ResNet18 with the specified SGD parameters including Nesterov.

#### Fine Tuning the Network

```
source ./stanford-car-classification-venv/bin/activate
python run_resnet_training_cli.py --path_images ./stanford-car-classification/data/ --path_data ./stanford-car-classification/data/cars_annos.mat --path_labels ./stanford-car-classification/data/cars_meta.mat --path_model_checkpoint ./stanford-car-classification --checkpoint_frequency 10 --model ResNet50 --epochs 35 --validation_frequency 10  --number_of_classes 196 --data_subset 1.0 --learning_rate_scheduler 0.2 --batch_size 6 --weight_decay 0.80 --momentum 0.9 --learning_rate 0.04 --scheduler_rate 5 --no-nesterov --no-freeze_weights

```

The above example fine tunes the whole ResNet50 architecture with the specified SGD parameters and no Nesterov.

## Hyperparameter Optimization for Tuning Pre-trained ResNet Models

The Hyperparameter Optimization (HPO) conducted is a layer on top of the model tuning as explicated above.

### Pre-requistes

The HPO uses [SigOpt](https://sigopt.com/)'s [Multitask](https://docs.sigopt.com/advanced_experimentation/multitask-experiments) feature as the optimizer as well as [Orchestrate](https://docs.sigopt.com/ai-module-api-references/orchestrate) to manage AWS clusters. 
In order to be able to run the optimization, please set up your SigOpt account and walk through the [Orchestrate tutorial](https://docs.sigopt.com/ai-module-api-references/orchestrate).
At the end of the tutorial, you should have an Orchestrate specific virtual environment which we will use later.

### CommandLine Interface

```
python run_resnet_training_cli.py
--path_images <path to parent directory of car_ims folder>
--path_data <path to cars_meta.mat> 
--path_labels <path to cars_annos.meta>
[--path_model_checkpoint <path to model checkpointing directory, default: No check-pointing>] 
[--checkpoint_frequency <frequency to generate PyTorch checkpoint files>, default: No check-pointing]
--model {ResNet18 | ResNet50}
--epochs <number of epochs to train model>
--validation_frequency <frequency to run validation during training> 
--number_of_classes <number of labels in data set>
--data_subset <subset of data to be used [0, 1.0]>
 {--freeze_weights | --no-freeze_weights}

```

To run HPO on tuning the fully connected layer: --freeze_weights must be included

To run HPO on fine tuning the whole network: --no-freeze_weights must be included

The data_subset option is used to specify the fraction of the Stanford Cars dataset to use (data_subset = 0.5 means 50% of the data is used).
A 20% validation split is applied to the data used.

Logs will be outputted to the working directory. We suggest to not checkpoint during hyperparameter optimization.

SigOpt suggests values for the following hyperparameters:
* batch size
* learning rate
* learning rate scheduler
* scheduler rate
* weight decay
* momentum
* nesterov

The bounds of these hyperparameters are specified in the Orchestrate experiment configuration file `orchestrate_stanford_cars_tuning_config.yml`.

Example:

```buildoutcfg
source ./stanford-car-classification-venv/bin/activate
python run_resnet_training_cli.py --path_images ./stanford-car-classification/data/ --path_data ./stanford-car-classification/data/cars_annos.mat --path_labels ./stanford-car-classification/data/cars_meta.mat --model ResNet18 --epochs 2 --validation_frequency 10 --data_subset 1.0  --number_of_classes 196 --no-freeze_weights
```

### Cluster Configuration

As seen in the [Orchestrate tutorial](https://docs.sigopt.com/ai-module-api-references/orchestrate/install_sigopt), a cluster configuration file is necessary to deploy a cluster.
The following snippet is an example cluster configuration `orchestrate_cluster_deploy_sample.yml` that deploys 2 p2.xlarge EC2 instances on AWS.

```buildoutcfg

# must be a .yml file
# AWS is currently our only supported provider for cluster create
provider: aws

# We have provided a name that is short and descriptive
cluster_name: stanford-cars-run-gpu-cluster

# Your cluster config can have CPU nodes, GPU nodes, or both.
# The configuration of your nodes is defined in the sections below.

# Define GPU compute here
gpu:
#   # AWS GPU-enabled instance type
#   # This can be any p* instance type
  instance_type: p2.xlarge
  max_nodes: 2
  min_nodes: 2

kubernetes_version: '1.12'

```

To see more options for AWS EC2 instances please read through AWS's EC2 [specification and pricing](https://aws.amazon.com/ec2/pricing/on-demand/).

### Orchestrate Experiment Configuration

The configuration file used to run SigOpt optimization using Orchestrate is in the repository as `orchestrate_stanford_cars_tuning_config.yml`.

Please note how the bounds for the hyperparameters are specified as well as the framework to be used and language.

### To Run The Hyperparameter Optimization

```buildoutcfg
source ./<ORCHESTRATE VENV>/bin/activate

sigopt cluster create -f cluster_deploy.yml

sigopt run -f orchestrate_stanford_cars_tuning_config.yml

```

To follow the progression of your job, look at the SigOpt dashboard or the following commands:

Status of all jobs in Orchestrate:

`sigopt status-all`

Status of a single job:

`sigopt status <job id>`

Status of a pod in the cluster:

`sigopt kubectl logs <pod name> -n orchestrate`
