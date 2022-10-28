import os
import json
import argparse
import time

from sigopt import Connection

from cnn_text.dataset import get_data
from cnn_text.objective import calculate_objective

parser = argparse.ArgumentParser(description='cnn text classification with SigOpt')
parser.add_argument('--with-architecture', default=False, action='store_true')
parser.add_argument('--experiment-id', type=int)

args = parser.parse_args()
with_architecture = args.with_architecture
experiment_id = args.experiment_id

# Instantiate Connection Object
SIGOPT_API_TOKEN = os.environ['SIGOPT_API_TOKEN']
conn = Connection(client_token=SIGOPT_API_TOKEN)

# Get experiment object
if experiment_id is None:

    # Get hyperparameters
    if with_architecture:
        exp_name = 'Multicriteria[failures] GPU-powered Sentiment Analysis (SGD + Architecture)'
        param_filepath = 'cnn_text/long_hyperparams.json'
    else:
        exp_name = 'Multicriteria[failures] GPU-powered Sentiment Analysis (SGD Only)'
        param_filepath='cnn_text/hyperparams.json'

    with open(param_filepath) as f:
        hyperparams = f.read()
        hyperparams = json.loads(hyperparams)

    experiment = conn.experiments().create(
        name=exp_name,
        parameters=hyperparams,
        observation_budget=30 * len(hyperparams),
        metrics=[
            {'name': 'accuracy', 'objective': 'maximize'},
            {'name': 'train_time', 'objective': 'minimize'},
        ],
    )

    print("Created experiment: https://sigopt.com/experiment/" + experiment.id)
else:
    experiment = conn.experiments(experiment_id).fetch()

# Optimization Loop
data = get_data()

def create_observation_dict(suggestion):
    start = time.time()
    accuracy = calculate_objective(
        suggestion.assignments,
        data,
        with_architecture=with_architecture,
    )
    end = time.time()

    failed = True
    values = None
    duration = end - start
    if accuracy > 75 and duration < 250:
        values = [
            {'name': 'accuracy', 'value': accuracy},
            {'name': 'train_time', 'value': duration},
        ]
        failed = False
    return {
        'suggestion': suggestion.id,
        'values': values,
        'failed': failed,
    }

for _ in range(experiment.observation_budget):
    suggestion = conn.experiments(experiment.id).suggestions().create()
    observation = conn.experiments(experiment.id).observations().create(**create_observation_dict(suggestion))
