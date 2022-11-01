
# Classifying the Stanford Cars Dataset

This repository runs hyperparameter optimization on tuning pretrained models from the [PyTorch model zoo](https://pytorch.org/docs/stable/torchvision/models.html) to classify images of cars in the Stanford Cars dataset.
This repository offers the option to tune only the fully connected layer of the pretrained network or fine tune the whole network.
The pretrained models supported are Resnet 18 and ResNet 50 are trained on ImageNet-1000.
This repository also offers the option to augment images through black box optimization and store these images either locally on disk or on Amazon S3.

## Quick look

### Quick overview

The following CLI supports:
* ResNet model training (from a pretrained start)
  * with Image Augmentation (in preprocessing)
  * without Image Augmentation (will only use the original Stanford Cars dataset images)
* ResNet model tuning (requires [Orchestrate](https://docs.sigopt.com/ai-module-api-references/orchestrate))
  * with Image Augmentation (augmentation will be included in the tuning loop)
  * without Image Augmentation (will tune the model off of the original dataset)

### Quick Links

* [CLI](#cli_training_no_image_aug) for training ResNet without Image Augmentation
* [CLI](#cli_training_with_image_aug) for training ResNet with Image Augmentation
* [Orchestrate](https://docs.sigopt.com/ai-module-api-references/orchestrate) [run config](./orchestrate_stanford_cars_tuning_config.yml) for hyperparameter tuning without Image Augmentation
* [Orchestrate](https://docs.sigopt.com/ai-module-api-references/orchestrate) [run config](./orchestrate_stanford_cars_augmented_tuning_config.yml) for hyperparameter tuning with Image Augmentation

### Quick Reads

* Post on [Insights for Building High-Performing Image Classification Models](https://mlconf.com/blog/insights-for-building-high-performing-image-classification-models/)

## Getting Started

### Clone repository

```buildoutcfg

git clone https://github.com/sigopt/stanford-augmented-car-classification

mkdir -p stanford-augmented-car-classification/data

```

### Download data

The Stanford Cars dataset can be found [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

The dataset includes:
* images of cars (cars_ims.tgz)
* labels (cars_annos.mat)
* devkit including human readable labels (cars_meta.mat)

```buildoutcfg

wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz -P stanford-augmented-car-classification/data

wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat -P stanford-augmented-car-classification/data

wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz -P stanford-augmented-car-classification/data

```

or using CURL:

```buildoutcfg
curl http://imagenet.stanford.edu/internal/car196/car_ims.tgz -o stanford-augmented-car-classification/data/car_ims.tgz

curl http://imagenet.stanford.edu/internal/car196/cars_annos.mat -o stanford-augmented-car-classification/data/cars_annos.mat

curl https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz -o stanford-augmented-car-classification/data/car_devkit.tgz

```

Unzip folders:

```buildoutcfg
tar -C stanford-augmented-car-classification/data -xzvf stanford-augmented-car-classification/data/car_ims.tgz
tar -C stanford-augmented-car-classification/data -xzvf stanford-augmented-car-classification/data/car_devkit.tgz

Only keeping the meta file in the devkit:

mv ./stanford-augmented-car-classification/data/devkit/cars_meta.mat ./stanford-augmented-car-classification/data
rm -r ./stanford-augmented-car-classification/data/devkit

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

python3 -m virtualenv ./stanford-augmented-car-classification-venv

```
#### Installing requirements in virtualenvironment

Use the `stanford_cars_venv_requirements.txt` file in this repository to install requirements to your virtualenvironment.

```buildoutcfg
source [PATH TO VIRTUALENV]/bin/activate (ex: ./stanford-augmented-car-classification-venv/bin/activate)

pip3 install -r stanford_cars_venv_requirements.txt

```

## Training Pre-trained ResNet Models

### <a name='cli_training_no_image_aug'></a>Training Pre-trained ResNet Models without Image Augmentation

#### CommandLine Interface

```
python resnet_stanford_cars_training.py --path_images <path to parent directory of car_ims folder>
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

To not output any directories: do not include ```--path_model_checkpoint```

To fine tune the whole network: --no-freeze_weights must be included

The CLI arguments learning_rate and weight_decay must be included in log base e form. For example, to specify a learning rate of 0.1 the CLI argument will look like ```--learning_rate ```

The data_subset option is used to specify the fraction of the Stanford Cars dataset to use (data_subset = 0.5 means 50% of the data is used).
A 20% validation split is applied to the data used.

Logs will be outputted to the working directory.

An output directory with the format `<time since epoch in seconds>_model` will be created under the current working directory.
Checkpoints will be stored under `<time since epoch in seconds>_model/model_checkpoints`.

##### Tuning the Fully Connected Layer Only

Example:

```
source ./stanford-augmented-car-classification-venv/bin/activate
python resnet_stanford_cars_training.py --path_images ./stanford-augmented-car-classification/data/ --path_data ./stanford-augmented-car-classification/data/cars_annos.mat --path_labels ./stanford-augmented-car-classification/data/cars_meta.mat --path_model_checkpoint ./stanford-augmented-car-classification --checkpoint_frequency 10 --model ResNet18 --epochs 35 --validation_frequency 10  --number_of_classes 196 --data_subset 1.0 --learning_rate_scheduler 0.2 --batch_size 6 --weight_decay 0.80 --momentum 0.9 --learning_rate 0.04 --scheduler_rate 5 --nesterov --freeze_weights

```

The above example tunes the fully connected layer of a pretrained ResNet18 with the specified SGD parameters including Nesterov.

##### Fine Tuning the Network

```
source ./stanford-augmented-car-classification-venv/bin/activate
python resnet_stanford_cars_training.py --path_images ./stanford-augmented-car-classification/data/ --path_data ./stanford-augmented-car-classification/data/cars_annos.mat --path_labels ./stanford-augmented-car-classification/data/cars_meta.mat --path_model_checkpoint ./stanford-augmented-car-classification --checkpoint_frequency 10 --model ResNet50 --epochs 35 --validation_frequency 10  --number_of_classes 196 --data_subset 1.0 --learning_rate_scheduler 0.2 --batch_size 6 --weight_decay 0.80 --momentum 0.9 --learning_rate 0.04 --scheduler_rate 5 --no-nesterov --no-freeze_weights

```

###<a name='cli_training_with_image_aug'></a>Training Pre-trained ResNet Models with Image Augmentation

#### CommandLine Interface

```
python resnet_stanford_cars_augmented_training.py
 --path_images <path to parent directory of car_ims folder>
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
--brightness <brightness factor to apply for image transformation [0, 10]>
--contrast <contrast factor to apply for image transformation [0, 100]>
--saturation <saturation factor to apply for image transformation [0, 100]>
--hue <hue factor to apply for image transformation [-0.5, 0.5]>
--multiplier <number of times to augment a singular image>
{--store_to_disk| --store_to_s3}
--probability <probability used for horizontal flip transformation
--s3_bucket_name <AWS S3 bucket name to store augmented images>

```

To include Nesterov in the learning: --nesterov must be included

To not include Nesterov: --no-nesterov must be included

To train the fully connected layer: --freeze_weights must be included

To fine tune the whole network: --no-freeze_weights must be included

To store on AWS S3 bucket: --store_to_s3

To store on disk: --store_to_disk

To not output any directories: do not include ```--path_model_checkpoint```

The CLI arguments learning_rate and weight_decay must be included in log base e form. For example, to specify a learning rate of 0.1 the CLI argument will look like ```--learning_rate ```

The data_subset option is used to specify the fraction of the Stanford Cars dataset to use (data_subset = 0.5 means 50% of the data is used).
A 20% validation split is applied to the data used.

Logs will be outputted to the working directory.

An output directory with the format `<time since epoch in seconds>_model` will be created under the current working directory.
Checkpoints will be stored under `<time since epoch in seconds>_model/model_checkpoints`.

##### Tuning the Fully Connected Layer Only with Image Augmentation

Example:

```
source ./stanford-augmented-car-classification-venv/bin/activate
python resnet_stanford_cars_augmented_training.py --path_images ./stanford-augmented-car-classification/data/ --path_data ./stanford-augmented-car-classification/data/cars_annos.mat --path_labels ./stanford-augmented-car-classification/data/cars_meta.mat --path_model_checkpoint ./stanford-augmented-car-classification --checkpoint_frequency 10 --model ResNet18 --epochs 35 --validation_frequency 10  --number_of_classes 196 --data_subset 1.0 --learning_rate_scheduler 0.2 --batch_size 6 --weight_decay 0.80 --momentum 0.9 --learning_rate 0.04 --scheduler_rate 5 --nesterov --freeze_weights --multiplier 2 --probability 1.0 --saturation 10 --hue -0.01 --brightness 20 --contrast 20 --store_to_s3 --s3_bucket_name stanford_cars_aug_bucket

```

The above example tunes the fully connected layer of a pretrained ResNet18 with the specified SGD parameters including Nesterov. Each image will be augmented twice and stored on AWS S3 in a bucket called stanford_cars_aug_bucket. The augmentation transformations will be applied as specified above.

##### Fine Tuning the Network with Image Augmentation

```
source ./stanford-augmented-car-classification-venv/bin/activate
python resnet_stanford_cars_augmented_training.py --path_images ./stanford-augmented-car-classification/data/ --path_data ./stanford-augmented-car-classification/data/cars_annos.mat --path_labels ./stanford-augmented-car-classification/data/cars_meta.mat --path_model_checkpoint ./stanford-augmented-car-classification --checkpoint_frequency 10 --model ResNet50 --epochs 35 --validation_frequency 10  --number_of_classes 196 --data_subset 1.0 --learning_rate_scheduler 0.2 --batch_size 6 --weight_decay 0.80 --momentum 0.9 --learning_rate 0.04 --scheduler_rate 5 --no-nesterov --no-freeze_weights --multiplier 1 --probability 1.0 --saturation 10 --hue -0.01 --brightness 20 --contrast 20 --store_to_disk

```

The above example fine tunes the whole ResNet50 architecture with the specified SGD parameters and no Nesterov. Each image will be augmented once and stored on disk. The augmentation transformations will be applied as specified above.

## Hyperparameter Optimization for Tuning Pre-trained ResNet Models

The Hyperparameter Optimization (HPO) conducted is a layer on top of the model tuning as explicated above. This repo currently supports hyperparameter optimization through SigOpt's implementation of Multitask and Orchesrate.

### Pre-requistes

The HPO uses [SigOpt](https://sigopt.com/)'s [Multitask](https://docs.sigopt.com/advanced_experimentation/multitask-experiments) feature as the optimizer as well as [Orchestrate](https://docs.sigopt.com/ai-module-api-references/orchestrate) to manage AWS clusters.
In order to be able to run the optimization, please set up your SigOpt account and walk through the [Orchestrate tutorial](https://docs.sigopt.com/ai-module-api-references/orchestrate/tutorial/1).
At the end of the tutorial, you should have an Orchestrate specific virtual environment which we will use later. This virtual environment is not the same as the virtual environment we set up above, and will include Orchestrate specific dependencies.

### CommandLine Interface

This command line interface will be used in the run configuration file for Orchestrate. Similar to the CLI for training the model, the CLI for optimization will support hyperparameter tuning with and without image augmentation.

Logs will be outputted to the working directory. We suggest to not checkpoint during hyperparameter optimization.

SigOpt suggests values for the following hyperparameters for model training:
* batch size
* learning rate
* learning rate scheduler
* scheduler rate
* weight decay
* momentum
* nesterov

SigOpt suggests values for the following hyperparameters for image augmentation:
* saturation
* hue
* contrast
* brightness

#### Hyperparameter Tuning without Image Augmentation CLI

```
python orchestrate_stanford_cars_cli.py --path_images <path to parent directory of car_ims folder>
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

To run HPO on tuning the fully connected layer: --freeze_weights must be included

To run HPO on fine tuning the whole network: --no-freeze_weights must be included

The data_subset option is used to specify the fraction of the Stanford Cars dataset to use (data_subset = 0.5 means 50% of the data is used).
A 20% validation split is applied to the data used.

The bounds of these hyperparameters are specified in the Orchestrate experiment configuration file `orchestrate_stanford_cars_tuning_config.yml`.

#### Hyperparameter Tuning with Image Augmentation CLI

```
python orchestrate_stanford_cars_augmentation_cli.py
 --path_images <path to parent directory of car_ims folder>
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
--brightness <brightness factor to apply for image transformation [0, 10]>
--contrast <contrast factor to apply for image transformation [0, 100]>
--saturation <saturation factor to apply for image transformation [0, 100]>
--hue <hue factor to apply for image transformation [-0.5, 0.5]>
--multiplier <number of times to augment a singular image>
{--store_to_disk| --store_to_s3}
--probability <probability used for horizontal flip transformation
--s3_bucket_name <AWS S3 bucket name to store augmented images>

```

To run HPO on tuning the fully connected layer: --freeze_weights must be included

To run HPO on fine tuning the whole network: --no-freeze_weights must be included

The data_subset option is used to specify the fraction of the Stanford Cars dataset to use (data_subset = 0.5 means 50% of the data is used).
A 20% validation split is applied to the data used.

The bounds of these hyperparameters are specified in the Orchestrate experiment configuration file `orchestrate_stanford_cars_augmented_tuning_config.yml`. Make sure to change the config file to include the name of your AWS S3 bucket used to store the augmented images.

### Cluster Configuration

As seen in the [Orchestrate tutorial](https://docs.sigopt.com/ai-module-api-references/orchestrate/tutorial/1), a cluster configuration file is necessary to deploy a cluster.
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

```

To see more options for AWS EC2 instances please read through AWS's EC2 [specification and pricing](https://aws.amazon.com/ec2/pricing/on-demand/).

### Orchestrate Experiment Configuration

Two Orchestrate run configuration files are provided in this repository:

* `orchestrate_stanford_cars_tuning_config.yml` - run hyperparameter tuning without image augmentation

* `orchestrate_stanford_cars_augmented_tuning_config.yml` - run hyperparameter tuning with image augmentation

Please note how the bounds for the hyperparameters are specified as well as the framework to be used and language.

### To Run The Hyperparameter Optimization with Image Augmentation

```buildoutcfg
source ./<ORCHESTRATE VENV>/bin/activate

sigopt cluster create -f cluster_deploy.yml

sigopt run -f orchestrate_stanford_cars_augmented_tuning_config.yml

```

Note: Make sure to change `orchestrate_stanford_cars_augmented_tuning_config.yml`  to include the name of your AWS S3 bucket used to store the augmented images.

To follow the progression of your job, look at the SigOpt dashboard or the following commands:

Status of all jobs in Orchestrate:

`sigopt status-all`

Status of a single job:

`sigopt status <job id>`

Status of a pod in the cluster:

`sigopt kubectl logs <pod name> -n orchestrate`
