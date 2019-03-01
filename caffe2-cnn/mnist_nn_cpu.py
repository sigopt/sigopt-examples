import csv
import math
import numpy as np
import os
import shutil
import subprocess
import urllib2
from caffe2.python import workspace, core, cnn, net_drawer, visualize
from sigopt import Connection

current_folder = os.path.join(os.getcwd(), 'caffe2_mnist')

data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
image_file_train = os.path.join(data_folder, "train-images-idx3-ubyte")
label_file_train = os.path.join(data_folder, "train-labels-idx1-ubyte")
image_file_test = os.path.join(data_folder, "t10k-images-idx3-ubyte")
label_file_test = os.path.join(data_folder, "t10k-labels-idx1-ubyte")

conn = Connection(client_token='SIGOPT_API_TOKEN')

# Get the dataset if it is missing
def download_dataset(url, path):
  import requests, zipfile, StringIO
  r = requests.get(url, stream=True)
  z = zipfile.ZipFile(StringIO.StringIO(r.content))
  z.extractall(path)

# Load training/testing data into db
def generate_db(image, label, name):
  name = os.path.join(data_folder, name)
  if not os.path.exists(name):
    syscall = "/usr/local/binaries/make_mnist_db --channel_first --db leveldb --image_file " + image + " --label_file " + label + " --output_file " + name
    subprocess.check_call(syscall, shell=True)

def data_setup():
  if not os.path.exists(data_folder):
    os.makedirs(data_folder)
  if not os.path.exists(label_file_train):
    download_dataset("https://s3.amazonaws.com/caffe2/datasets/mnist/mnist.zip", data_folder)

  if os.path.exists(root_folder):
    shutil.rmtree(root_folder)

  os.makedirs(root_folder)
  workspace.ResetWorkspace(root_folder)

  # (Re)generate the leveldb database (known to get corrupted...)
  generate_db(image_file_train, label_file_train, "mnist-train-nchw-leveldb")
  generate_db(image_file_test, label_file_test, "mnist-test-nchw-leveldb")

# Define the SigOpt experiment
def setup_sigopt_experiment(conn):
  experiment = conn.experiments().create(
    name='MNIST Dataset Deep Neural Net (Caffe2)',
    project='sigopt-examples',
    observation_budget=40,
    parameters=[
      {
        'name': 'conv1_dim',
        'type': 'int',
        'bounds': {
          'min': 5,
          'max': 50,
        }
      },
      {
        'name': 'conv2_dim',
        'type': 'int',
        'bounds': {
          'min': 5,
          'max': 100,
        }
      },
      {
        'name': 'log_learning_rate',
        'type': 'double',
        'bounds': {
            'min': math.log(1e-7),
            'max': math.log(1),
        },
      },
      {
        'name': 'fc3',
        'type': 'int',
        'bounds': {
            'min': 10,
            'max': 1000,
        },
      },
    ]
  )
  return experiment

# Create the model given hyperparameter assignments
def create_model(assignments):
  model = cnn.CNNModelHelper(name="mnist_train")

  # Add data to model
  data_uint8, label = model.TensorProtosDBInput(
    [], ["data_uint8", "label"], batch_size=64,
    db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'), db_type='leveldb')
  data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
  data = model.Scale(data, data, scale=float(1./256))
  data = model.StopGradient(data, data)

  # Create neural network structure
  conv1 = model.Conv(data, 'conv1', dim_in=1, dim_out=assignments['conv1_dim'], kernel=5)
  pool1 = model.MaxPool(conv1, 'pool1', kernel=2, stride=2)
  conv2 = model.Conv(pool1, 'conv2', dim_in=assignments['conv1_dim'], dim_out=assignments['conv2_dim'], kernel=5)
  pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=2)
  fc3 = model.FC(pool2, 'fc3', dim_in=assignments['conv2_dim'] * 4 * 4, dim_out=assignments['fc3'])
  fc3 = model.Relu(fc3, fc3)
  pred = model.FC(fc3, 'pred', assignments['fc3'], 10)
  softmax = model.Softmax(pred, 'softmax')

  # Add Training Operators
  xent = model.LabelCrossEntropy([softmax, label], 'xent')
  loss = model.AveragedLoss(xent, "loss")
  accuracy = model.Accuracy([softmax, label], "accuracy")
  model.AddGradientOperators([loss])
  ITER = model.Iter("iter")
  LR = model.LearningRate(
    ITER, "LR", base_lr=-math.exp(assignments['log_learning_rate']), policy="step", stepsize=1, gamma=0.999 )
  ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
  for param in model.params:
    param_grad = model.param_to_grad[param]
    model.WeightedSum([param, ONE, param_grad, LR], param)

  return model

# Evaluates the model and returns the accuracy
def train_model(model):
  workspace.RunNetOnce(model.param_init_net)
  workspace.CreateNet(model.net)
  total_iters = 100
  for i in range(total_iters):
    workspace.RunNet(model.net.Proto().name)

  return workspace.FetchBlob('accuracy').item()

# Loops through retrieving suggestion, evaluating model, and reporting observation
def sigopt_optimization_loop(conn, experiment):
  for _ in range(experiment.observation_budget):
    # Fetch new suggestion
    suggestion = conn.experiments(experiment.id).suggestions().create()
    assignments = suggestion.assignments

    # Test suggestion
    model = create_model(assignments)
    accuracy = train_model(model)

    # Report observation
    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        value=accuracy
      )

  # Get most accurate assignments
  experiment = conn.experiments(experiment.id).fetch()
  assignments = conn.experiments(experiment.id).best_assignments().fetch()

  print("Best Assignments:")
  print(assignments.data[0].assignments)

  return assignments.data[0].assignments

def main():
  # Downloads data if it doesn't exist yet
  data_setup()

  # Create experiment in SigOpt portal
  experiment = setup_sigopt_experiment(conn)

  # Run the SigOpt optimization loop
  assignments = sigopt_optimization_loop(conn, experiment)

  # This is a SigOpt-tuned model
  create_model(assignments)

if __name__ == "__main__":
  main()
