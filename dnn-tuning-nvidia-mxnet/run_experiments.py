import os, json, argparse

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
        exp_name = 'GPU-powered Sentiment Analysis (SGD + Architecture)'
        param_filepath = 'cnn_text/long_hyperparams.json'
    else:
        exp_name = 'GPU-powered Sentiment Analysis (SGD Only)'
        param_filepath='cnn_text/hyperparams.json'

    with open(param_filepath) as f:
        hyperparams = f.read()
        hyperparams = json.loads(hyperparams)

    experiment = conn.experiments().create(
                         name=exp_name,
                         project='sigopt-examples',
                         metrics=[dict(name='value', objective='maximize')],
                         parameters=hyperparams,
                         observation_budget=40*len(hyperparams))

    print("Created experiment: https://sigopt.com/experiment/" + experiment.id)
else:
    experiment = conn.experiments(experiment_id).fetch()

# Optimization Loop
data = get_data()
for _ in range(experiment.observation_budget):
    suggestion = conn.experiments(experiment.id).suggestions().create()
    value = calculate_objective(suggestion.assignments, data,
                                with_architecture=with_architecture)
    observation = conn.experiments(experiment.id).observations().create(
                                value=value,
                                suggestion=suggestion.id)
