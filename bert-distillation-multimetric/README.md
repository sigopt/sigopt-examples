# optimizing-distilbert-squad

This repo supports code for tuning [DistilBERT](https://arxiv.org/abs/1910.01108) with [SigOpt's Multimetric Bayesian Optimization](https://app.sigopt.com/docs/overview/multimetric). Before using this repo, read through [HuggingFace's Transformer package](https://huggingface.co/transformers/) and reading the full blog post on this work for more context.

## Download SQUAD 2.0

This repo required SQUAD 2.0 to be locally downloaded.

Create a data directory:

`mkdir -p ./data`

Get [SQUAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) files.

```
wget -P ./data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

wget -P ./data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

wget -P ./data https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
```

Change name of evaluation script:

```
mv ./data/index.html ./data/evaluate-v2.0.py
```

## Setting up your virtualenvironment

Make a new Python3 virtualenvironment:

```
virtualenv -p $(which python3) ./venv
source venv/bin/activate
pip3 install transformers==2.4.1
pip3 install scikit-learn
pip3 install boto3
pip3 install tensorboard
pip3 install torch torchvision
pip3 install logbeam
pip3 install sigopt
pip3 install 'ray[tune]'
```

## CLI Options

You have 3 main ways to run the repo:

* Run distillation on Squad 2.0
* Optimize distillation on Squad 2.0
* Orchestrate optimization with Ray

### Run Distillation on Squad 2.0

First, let's walk through how to run the distillation without optimization.

Main CLI options:

* model_type: student model type
* teacher_type: teacher model type
* teacher_name_or_path: local checkpoints for teacher model or a model from [HuggingFace's model zoo](https://huggingface.co/models)
* train_file: path to SQUAD 2.0 training file
* predict_file: path to SQUAD 2.0 dev file
* output_dir: output directory for training outputs
* num_train_epochs: number of epochs to train student model during distillation
* cache_s3_bucket: flag to download stored caches in s3
* train_cache_s3_directory: location of training data's cached features on s3
* eval_cache_s3_directory: location of dev data's cached features in s3

Default CLI options:
* logging_steps: interval number of steps to log
* do_lower_case: loads dataset with all lowercase
* save_steps: interval number of steps to checkpoint
* version_2_with_negative: loads dataset with SQUAD 2.0 features

Default Parameters:
* alpha_ce: weight for soft target loss
* alpha_squad: weight for hard target loss
* temperature: temperature for distillation
* max_seq_length: max length for generating features from SQUAD 2.0
* max_query_length: max length for generating features for questions from SQUAD 2.0
* max_answer_length: used to calculate squad 2.0 metrics
* learning_rate: learning rate for student model
* weight_decay: weight decay for student model
* adam_epsilon: epsilon for Adam optimizer
* n_heads: number of multiattention heads in student model
* n_layers: number of Transformer layers in student model
* dropout: dropout for the network
* attention_dropout: dropout for the multiattention heads
* qa_dropout: dropout for question answering layer

For a full list of defaults and their values go to ```./sigopt-examples/bert-distillation-multimetric/distilbert_run_and_hpo_configurations/distilbert_squad_hpo_parameters.py```

Example CLI:

```
python sigopt-examples/bert-distillation-multimetric/squad_fine_tuning/run_squad_w_distillation.py --model_type distilbert --teacher_type bert --teacher_name_or_path bert-base-uncased --train_file ../data/SQUAD_2/train-v2.0_subset.json --predict_file ../data/SQUAD_2_subset/dev-v2.0_subset2.json --output_dir ../test_run --num_train_epochs 3
```
The above cli will run the distillation process for SQUAD 2.0 with the defaults listed above. The student mdoel is distilbert, the teacher model is bert, and it runs the student model training for 3 epochs.

### Optimize the distillation process

Run [SigOpt's Multimetric Bayesian Optimization](https://app.sigopt.com/docs/overview/multimetric) with distillation.  

Main CLI options:

* model_type: student model type
* teacher_type: teacher model type
* teacher_name_or_path: local checkpoints for teacher model or a model from [HuggingFace's model zoo](https://huggingface.co/models)
* train_file: path to SQUAD 2.0 training file
* predict_file: path to SQUAD 2.0 dev file
* output_dir: output directory for training outputs
* num_train_epochs: number of epochs to train student model during distillation
* experiment_name: SigOpt experiment name
* api_token: SigOpt API Token
* use_hpo_default_ranges: flag to use default hpo ranges specfied in ```sigopt-examples/bert-distillation-multimetric/sigopt_optimization_wrapper/sigopt_hyperparameter_definition.py```
* sigopt_experiment_id: experiment id of existing SigOpt experiment. if not None, will load existing experiment
* sigopt_observation_budget: number of optimization runs for SigOpt experiment
* store_s3: flag to store checkpoints to s3
* s3_bucket: s3 bucket name for checkpoint storing
* cache_s3_bucket: flag to download stored caches in s3
* train_cache_s3_directory: location of training data's cached features on s3
* eval_cache_s3_directory: location of dev data's cached features in s3


For a full list of default hyperparameter ranges, go to: ```sigopt-examples/bert-distillation-multimetric/sigopt_optimization_wrapper/sigopt_hyperparameter_definition.py```. And for a full list of defaults, go to: ```sigopt-examples/bert-distillation-multimetric/squad_distillation_abstract_clis/a_optimizaton_run_squad_cli.py```.

Example CLI:

```
python sigopt-examples/bert-distillation-multimetric/sigopt_optimization_cli.py --model_type distilbert --train_file ./data/train-v2.0.json
--predict_file ./data/dev-v2.0.json --experiment_name test-multimetric-distillation --project_name test-multimetric --use_hpo_default_ranges --api_token <SigOpt API Token> --output_dir
./test_run_5 --sigopt_observation_budget 200 --teacher_type bert --teacher_name_or_path twmkn9/bert-base-uncased-squad2 --num_train_epochs 3 --store_s3 --s3_bucket s3-checkpoint-bucket --cache_s3_bucket --train_cache_s3_directory s3-cache-bucket/train_cache --eval_cache_s3_directory s3-cache-bucket/dev_cache
```

The above CLI runs Multimetric Optimization on the distillation process using a [teacher model](https://huggingface.co/twmkn9/bert-base-uncased-squad2) from the [HuggingFace model zoo](https://huggingface.co/models). The optimization will run for 200 cycles and train the student model for 3 epochs each. The checkpoints will be stored on s3 and logged/saved for every 1000 steps. The feature caches for the dataset will be pulled from the specified s3 bucket.

## Using Ray to orchestrate the optimization process

Run [SigOpt's Multimetric Bayesian Optimization](https://app.sigopt.com/docs/overview/multimetric) with distillation using [Ray](https://docs.ray.io/en/master/) for orchestration.

Before using this CLI, look through the [Ray documentation](https://docs.ray.io/en/master/).

The following Ray orchestration will use this AMI to set up the environment for the nodes in the cluster.

To run the optimization process on a Ray cluster:

1. Launch a Ray cluster. There is an [example config](raytune_wrapper/ray_launch_config.yaml) in the repo

2. Execute the ray cli on the cluster

Main CLI options:

* model_type: student model type
* teacher_type: teacher model type
* teacher_name_or_path: local checkpoints for teacher model or a model from [HuggingFace's model zoo](https://huggingface.co/models)
* train_file: path to SQUAD 2.0 training file
* predict_file: path to SQUAD 2.0 dev file
* output_dir: output directory for training outputs
* num_train_epochs: number of epochs to train student model during distillation
* experiment_name: SigOpt experiment name
* api_token: SigOpt API Token
* use_hpo_default_ranges: flag to use default hpo ranges specfied in ```optimizing-distilbert-squad/sigopt_optimization_wrapper/sigopt_hyperparameter_definition.py```
* sigopt_experiment_id: experiment id of existing SigOpt experiment. if not None, will load existing experiment
* sigopt_observation_budget: number of optimization runs for SigOpt experiment
* store_s3: flag to store checkpoints to s3
* s3_bucket: s3 bucket name for checkpoint storing
* cache_s3_bucket: flag to download stored caches in s3
* train_cache_s3_directory: location of training data's cached features on s3
* eval_cache_s3_directory: location of dev data's cached features in s3
* max_concurrent: max number of concurrent workers used
* parallel: total number of parallel workers used for SigOpt
* num_cpu: number of cpus required for each run
* num_gpu: number of gpus required for each run
* ray_address: ip address of ray cluster's head node
* clean_raytune_output: flag to clear RayTune outputs after the run
* raytune_output_directory: output directory for RayTune

Example CLI:

```
python sigopt-examples/bert-distillation-multimetric/sigopt_ray_optimization_cli.py
        --model_type distilbert
	--teacher_type bert
	--teacher_name_or_path twmkn9/bert-base-uncased-squad2
	--train_file /home/ubuntu/SQUAD_2/train-v2.0.json
	--predict_file /home/ubuntu/SQUAD_2/dev-v2.0.json
	--num_train_epochs 3
	--experiment_name bert-distillation-full-run
	--project_name 	bert-distillation-full-run
        --use_hpo_default_ranges
        --api_token <SigOpt API Token>
        --output_dir /home/ubuntu/output_dir
	--overwrite_output_dir
	--sigopt_observation_budget 479
	--parallel 20
	--max_concurrent 20
	--num_cpu 8
	--num_gpu 1
	--store_s3
	--s3_bucket <S3_checkpoint_bucket>
	--ray_address <RAY_IP_ADDRESS>
	--cache_s3_bucket <S3_cache_bucket>
	--train_cache_s3_directory <S3_path_train_features>
        --eval_cache_s3_directory <S3_path_dev_features>
  ```
The above cli executes the optimization process across 20 workers in parallel.


## For questions and access to SigOpt
If you have any questions when running the repo, please feel free to reach out. [Contact us](https://sigopt.com/try-it) for access to SigOpt.
