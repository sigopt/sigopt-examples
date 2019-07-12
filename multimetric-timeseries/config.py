import os, json

import numpy as np

THIS_DIR = os.path.dirname(__file__)


with open(os.path.join(THIS_DIR, 'sigopt.secret')) as f:
    config = f.read()
    SIGOPT_API_TOKEN = json.loads(config)['SIGOPT_API_TOKEN']

PARAMETERS = [
  dict(name='batch_size', type='int', bounds={'min': 8,'max': 32}),
  dict(name='conv_1_num_filters',  type='int', bounds={'min': 32,'max': 256}),
  dict(name='conv_1_filter_size', type='int', bounds={'min': 2,'max': 10}),
  dict(name='conv_2_num_filters', type='int', bounds={'min': 32,'max': 256}),
  dict(name='conv_2_filter_size', type='int', bounds={'min': 2,'max': 10}),
  dict(name='conv_3_num_filters', type='int', bounds={'min': 32,'max': 256}),
  dict(name='conv_3_filter_size', type='int', bounds={'min': 2,'max': 10}),
  dict(name='log_lr', type='double', bounds={'min': np.log(1e-10),'max': np.log(1)}),
  dict(name='log_beta_1', type='double', bounds={'min': np.log(1e-2),'max': np.log(.5)}),
  dict(name='log_beta_2', type='double', bounds={'min': np.log(1e-6),'max': np.log(.5)}),
  dict(name='log_epsilon', type='double', bounds={'min': np.log(1e-10),'max': np.log(1e-6)}),
  dict(name='log_decay', type='double', bounds={'min': np.log(1e-10),'max': np.log(1e-1)}),
]

DATASET = 'Adiac'
DATASET_FOLDER = os.path.join('UCR_TS_Archive_2015', DATASET)
DATASET_FILE = DATASET
DATASET_PATH = os.path.join(THIS_DIR, DATASET_FOLDER, DATASET_FILE)

EXPERIMENT_NAME = 'multimetric time series accuracy vs. inference time'

PROJECT_NAME = 'sigopt-examples'

METRIC_1 = 'val_acc'
METRIC_2 = 'negative_inference_time'

METRICS = [
  {'name': METRIC_1, 'objective': 'maximize'},
  {'name': METRIC_2, 'objective': 'maximize'},
]

NB_EPOCHS = 500

OBSERVATION_BUDGET = len(PARAMETERS) * 20 * 2
