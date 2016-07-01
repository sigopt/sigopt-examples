'''
This script is written for use with AWS AMI : ami-d7562bb7 - Nervana neon and ncloud
The AMI is available in N. California Region in the Community AMI listings
Recommend running using a g2.2xlarge instance
Before running this script please source the virtual env and download the cifar dataset :
source ./neon/.venv/bin/activate
pip install sigopt
./neon/neon/data/batch_writer.py --set_type cifar10 --data_dir "/home/ubuntu/data" --macro_size 10000 --target_size 40
'''

import logging, sys
import datetime
import sigopt.interface
import time
from neon.initializers import Gaussian
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import DataIterator, load_cifar10
from neon.callbacks.callbacks import Callbacks, MetricCallback
from neon.util.argparser import NeonArgparser
from sigopt_creds import client_token


parser = NeonArgparser(__doc__)
args = parser.parse_args()

conn = sigopt.interface.Connection(client_token=client_token)
experiment = conn.experiments().create(
  name='Nervana All CNN GPU '+datetime.datetime.now().strftime("%Y_%m_%d_%I%M_%S"),
  parameters=[
    { "name": "log(learning_rate)",   "type": "double", "bounds": {"max": -0.3,  "min": -3.0,}},
    { "name": "log(weight_decay)",    "type": "double", "bounds": {"max": 0.0,   "min": -3.0,}},
    { "name": "gaussian_scale",       "type": "double", "bounds": {"max": 0.5,   "min": 0.01,}},
    { "name": "momentum_coef",        "type": "double", "bounds": {"max": 0.999, "min": 0.001,}},
    { "name": "momentum_step_change", "type": "double", "bounds": {"max": 0.999, "min": 0.001,}},
    { "name": "momentum_step_schedule_start","type": "int", "bounds": { "min": 50, "max": 300,}},
    { "name": "momentum_step_schedule_step_width","type": "int", "bounds": {"max": 100,"min": 5,}},
    { "name": "momentum_step_schedule_steps", "type": "int", "bounds": {"max": 20,"min": 1,}},
    { "name": "epochs","type": "int", "bounds": {"max": 500,"min": 50,}},
  ],
  observation_budget=180,
)

DATA_DIR = "/home/ubuntu/data"

(X_train, y_train), (X_test, y_test), nclass = load_cifar10(
    path=DATA_DIR,
    normalize=False,
    contrast_normalize=True,
    whiten=False,
    )

# get error on this command
train_set = DataIterator(X_train, y_train, nclass=16, lshape=(3, 32, 32))
valid_set = DataIterator(X_test, y_test, nclass=16, lshape=(3, 32, 32))

# run optimization loop
for ir in xrange(experiment.observation_budget):
  suggestion = conn.experiments(experiment.id).suggestions().create()
  assignments = suggestion.assignments
  print assignments

  num_epochs = int(assignments.get("epochs"))
  init_uni = Gaussian(scale=assignments.get("gaussian_scale"))
  step_config = [int(assignments.get("momentum_step_schedule_start") + i*assignments.get("momentum_step_schedule_step_width")) for i in range(int(assignments.get("momentum_step_schedule_steps")))]
  opt_gdm = GradientDescentMomentum(
    	learning_rate=float(10.0**assignments.get("log(learning_rate)")),
    	momentum_coef=float(assignments.get("momentum_coef")),
    	wdecay=float(10.0**assignments.get("log(weight_decay)")),
    	schedule=Schedule(step_config=step_config, change=float(assignments.get("momentum_step_change"))),
    )

  relu = Rectlin()
  conv = dict(init=init_uni, batch_norm=False, activation=relu)
  convp1 = dict(init=init_uni, batch_norm=False, activation=relu, padding=1)
  convp1s2 = dict(init=init_uni, batch_norm=False, activation=relu, padding=1, strides=2)

  layers = [Dropout(keep=.8),
            Conv((3, 3, 96), **convp1),
            Conv((3, 3, 96), **convp1),
            Conv((3, 3, 96), **convp1s2),
            Dropout(keep=.5),
            Conv((3, 3, 192), **convp1),
            Conv((3, 3, 192), **convp1),
            Conv((3, 3, 192), **convp1s2),
            Dropout(keep=.5),
            Conv((3, 3, 192), **convp1),
            Conv((1, 1, 192), **conv),
            Conv((1, 1, 16), **conv),
            Pooling(8, op="avg"),
            Activation(Softmax())]

  cost = GeneralizedCost(costfunc=CrossEntropyMulti())

  mlp = Model(layers=layers)

  # configure callbacks
  callbacks = Callbacks(mlp)

  def do_nothing(_):
    pass

  callbacks.callbacks = []
  callbacks.on_train_begin = do_nothing
  callbacks.on_epoch_end = do_nothing

  mlp.fit(train_set, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)
  opt_metric = 1.0 - mlp.eval(valid_set, metric=Misclassification())
  print('Metric = {}'.format(opt_metric))
  conn.experiments(experiment.id).observations().create(
    suggestion=suggestion.id,
    value=float(opt_metric[0]),
  )
