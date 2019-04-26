# Optimizing MemN2N

Implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) with hyperparameter optimization. Tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset. Original GitHub repo can be found [here](https://github.com/domluna/memn2n).

## Requirements

* tensorflow 1.0.0
* scikit-learn 0.17.1
* six 1.10.0

For requirements information and installation instructions see [here](./virtual_env_setup/README.md).

## Get Started

```
git clone https://github.com/sigopt/optimizing-memn2n.git

mkdir ./end2end_mem_nn_tensorflow/data/

cd ./end2end_mem_nn_tensorflow/data/
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
(or if you don't have wget try:
curl http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz -o tasks_1-20_v1-2.tar.gz)
tar xzvf ./tasks_1-20_v1-2.tar.gz

cd ../
source <[Path to virtual environment]>/bin/activate
python sigopt_optimization_run.py --run_single_exp True --run_joint_exp False --task_id 20 --sigopt_observation_budget 4 --sigopt_connection_token <SIGOPT API TOKEN> --sigopt_experiment_name 'task 20 conditionals optimization' --experiment_type 'conditionals'
```

## Command Line Interface

`python sigopt_optimization_run.py [--log_file <file path>] {--run_single_exp --task_id <task id> --sigopt_calc_accuracy_tasks None | --run_joint_exp --sigopt_calc_accuracy_tasks <string of comma separated task ids> --task_id None} [--max_grad_norm <l2 norm clipping>] [--batch_size <batch size>] [--epochs <total number of epochs>] [--random_state <random seed>] [--data_dir <path to bAbI data directory>] --sigopt_connection_token <API token for Sigopt account> {--sigopt_observation_budget <number of Sigopt observations> --sigopt_experiment_name <Sigopt experiment name> --experiment_type {random|sigopt|conditionals} | --sigopt_experiment_id <existing Sigopt experiment id>}`  

If optional arguments are not provided, the above command line will use the following default parameters and assumes the working directory is ./end2end_mem_nn_tensorflow:

* log_file = [WORKING DIRECTORY]/memn2n_optimization_run.log
* max_grad_norm = 40
* batch_size = 32
* epochs = 60
* random_state = None
* data_dir = ./data/tasks_1-20_v1-2/en/ (English 1k dataset)
* sigopt_experiment_id = None

Validation is performed on the validation and test sets of the data at the end of the specified number of epochs. The test accuracy calculated (either averaged over 20 tasks for joint training or specified task for single training) at this point is reported back as the optimization metric to Sigopt.

### Running hyperparameter optimization for single task training

#### Example: Task 20 optimization on English 1k dataset

`python sigopt_optimization_run.py --log_file '/Users/username/memory_network_opt.log' --run_single_exp --task_id 20 --sigopt_observation_budget 4 --sigopt_connection_token 'OEIUROEAHE889823I' --sigopt_experiment_name 'task 20 conditionals optimization' --experiment_type 'conditionals'`

The above command will optimize bAbI task 20 using [Sigopt Conditionals](https://app.sigopt.com/docs/overview/conditionals) for 4 [observation cycles](https://app.sigopt.com/docs/overview/optimization) using default parameters for `max_grad_norm, batch_size, epochs, random_state, and data_dir`. Must be run with end2end_mem_nn_tensorflow as the working directory.

#### Example: Task 1 optimization

`python sigopt_optimization_run.py --log_file '/Users/username/memory_network_opt.log' --run_single_exp --task_id 20 --max_grad_norm 10 --batch_size 31 --epochs 10 --data_dir '/Users/username/Downloads/tasks_1-20_v1-2/hn' --sigopt_observation_budget 2 --sigopt_connection_token 'OEIUROEAHE889823I' --sigopt_experiment_name 'task 1 random optimization' --experiment_type random`

The above command will optimize bAbI task 1 using Sigopt's Random search implementation for 2 observation cycles with the specified max_grad_norm, batch_size, and epochs.

#### Example: Task 4 optimization with existing experiment

`python sigopt_optimization_run.py --log_file '/Users/username/memory_network_opt.log' --run_single_exp --task_id 4 --sigopt_connection_token 'OEIUROEAHE889823I' --sigopt_experiment_id 67770`

The above command will pick up an existing/already created experiment and continue to optimize until it meets the predefined number of optimization cycles. This can be found under the Experiment [properties page](https://app.sigopt.com/experiment/43973/properties). Must be run with end2end_mem_nn_tensorflow as the working directory. The task id specified on the CLI must match the task id previously specified for the experiment.

### Running hyperparameter optimization for joint training

#### Example: Joint training on 1k English dataset

`python sigopt_optimization_run.py --log_file '/Users/username/memory_network_opt.log' --run_joint_exp --sigopt_calc_accuracy_tasks '7,8,10' --max_grad_norm 10 --batch_size 31 --epochs 10 --sigopt_observation_budget 200 --sigopt_connection_token 'OEIUROEAHE889823I' --sigopt_experiment_name '20 task sigopt optimization' --experiment_type sigopt`

The above command will run optimization on all 20 tasks using Sigopt's Bayesian Optimization for 200 observation cycles. For each Sigopt observation cycle, it will store the test accuracies for tasks 7,8,10 as [experiment metadata](https://app.sigopt.com/docs/objects/metadata). Must be run with end2end_mem_nn_tensorflow as the working directory.

#### Example: Joint training with existing experiment

`python sigopt_optimization_run.py --log_file '/Users/username/memory_network_opt.log' --run_joint_exp --sigopt_calc_accuracy_tasks '7,8,10' --sigopt_connection_token 'OEIUROEAHE889823I' --sigopt_experiment_id 67773`

The above command will pick up an existing experiment and continue to optimize until it meets the predefined number of optimization cycles. Must be run with end2end_mem_nn_tensorflow as the working directory.

## Instructions to setup EC2 instances

### Launch a p2.xlarge instance

Using the AWS dashboard, choose the latest [Ubuntu Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B077GCH38C).

Select to launch the GPU compute p2.xlarge instance.

For this repo to run with GPU, it requires Tensorflow-gpu 1.0.0, Cuda 8.0, and CudNN 5.1.

#### Download CudNN 5.1

You will need a Nvidia developer's login to download the [archived CudNN version](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod_20161129/8.0/cudnn-8.0-linux-x64-v5.1-tgz). Once you have the archived on your instance run the following commands:

unzip cudnn package:

`tar -xvzf ./cudnn-8.0-linux-x64-v5.1-tgz`

#### Remove existing symlink between cuda 9.0 and cuda:

`rm -rf /usr/local/cuda`

`sudo rm /usr/local/cuda8.0/lib64/libcudnn*`

#### Replace existing cudNN libraries with cudNN 5.1.0

copy cudnn to cuda8.0:

`sudo cp -P ./cuda/include/cudnn.h /usr/local/cuda8.0/include`

copy cudnn libraries to cuda8:

`sudo cp -P ./cuda/lib64/libcudnn* /usr/local/cuda8.0/lib64`

create symlink from cuda8 to cuda:

`ln -s /usr/local/cuda8.0 /usr/local/cuda`

Now the Cuda version on your instance will be set to 8.0 and the cuDNN will be 5.1.0.

### Creating a compatible python environment

#### Install pip3

`sudo apt-get install python3-pip`

#### Install virtualenv

`pip3 install virtualenv`

#### Create a new virtualenv and activate

`virtualenv -p $(which python3) <LOCATION OF ENVIRONMENT>`

`source <LOCATION OF ENVIRONMENT>/bin/activate`

#### Copy virtualenv gpu requirements to launched instance(s)

`scp -i <LOCATION OF PEM KEY> [...]/optimizing-memn2n/virtualenv_env_setup/python3_memn2n_tf_env_gpu_requirements.txt ubuntu@<INSTANCE PUBLIC DNS>:~`

#### For GPU compatible environment:

`pip install -r [...]/python3_memn2n_tf_env_gpu_requirements.txt`

As we activated our virtual environment in a previous step, these requirements will be installed within the environment.

#### Copy and unzip repo on instances

`scp -i <LOCATION OF PEM KEY> [...]/optimizing-memn2n.zip ubuntu@<INSTANCE PUBLIC DNS>:~`

`ssh -i <LOCATION OF PEM KEY> ubuntu@<INSTANCE PUBLIC DNS>`

`unzip optimizing-memn2n.zip`

### Activate instance and run optimization

`source <LOCATION OF PEM KEY>/bin/activate`

`python sigopt_optimization_run.py --log_file '/home/ubuntu/memory_network_opt.log' --data_dir '/home/ubuntu/data/tasks_1-20_v1-2/en' --run_single_exp False --run_joint_exp True --sigopt_calc_accuracy_tasks '7,8,10' --max_grad_norm 10 --batch_size 31 --epochs 10 --sigopt_observation_budget 200 --sigopt_connection_token 'OEIUROEAHE889823I' --sigopt_experiment_name '20 task sigopt optimization' --experiment_type sigopt`
