from collections import namedtuple

# Parameter search constants
NUM_SIGOPT_SUGGESTIONS = 40
GRID_SEARCH_WIDTH = 4
NUM_RANDOM_SEARCHES = 192

Dataset = namedtuple(
        'Dataset',
        [
            'X_train',
            'y_train',
            'X_test',
            'y_test',
            ],
        )

# ML hyperparameter spaces
# Email contact@sigopt.com if you need help
# Examples at http://sigopt.com/docs

SVC_PARAMS = [
            {
                "bounds": {
                    "max": 10.0,
                    "min": 0.01,
                    },
                "name": "C",
                "type": "double",
                "transformation": "log",
                },
            {
                "bounds": {
                    "max": 1.0,
                    "min": 0.0001,
                    },
                "name": "gamma",
                "type": "double",
                "transformation": "log",
                },
            {
                "type": "categorical",
                "name": "kernel",
                "categorical_values": [
                    {"name": "rbf"},
                    {"name": "poly"},
                    {"name": "sigmoid"},
                    ],
                },
            ]

GBC_PARAMS = [
            {
                "bounds": {
                    "max": 1.0,
                    "min": 0.01,
                    },
                "name": "learning_rate",
                "type": "double",
                "transformation": "log",
                },
            {
                "bounds": {
                    "max": 500,
                    "min": 20,
                    },
                "name": "n_estimators",
                "type": "int",
                "transformation": "log",
                },
            {
                "bounds": {
                    "max": 4,
                    "min": 1,
                    },
                "name": "min_samples_split",
                "type": "int",
                },
            {
                "bounds": {
                    "max": 3,
                    "min": 1,
                    },
                "name": "min_samples_leaf",
                "type": "int",
                },
            ]

RFC_PARAMS = [
            {
                "bounds": {
                    "max": 3,
                    "min": 1,
                    },
                "name": "min_samples_leaf",
                "type": "int",
                },
            {
                "bounds": {
                    "max": 20,
                    "min": 3,
                    },
                "name": "n_estimators",
                "type": "int",
                },
            {
                "bounds": {
                    "max": 4,
                    "min": 1,
                    },
                "name": "min_samples_split",
                "type": "int",
                },
            {
                "bounds": {
                    "max": 1.0,
                    "min": 0.1,
                    },
                "name": "max_features",
                "type": "double",
                },
            ]

CLASSIFIER_TYPE_TO_PARAMS = {
        'GBC': GBC_PARAMS,
        'SVC': SVC_PARAMS,
        'RFC': RFC_PARAMS,
        }
