#!/usr/bin/env bash
######################################################################
# This script installs MXNet for Python along with all required dependencies on a Amazon Linux Machine.
######################################################################
set -ex

sudo apt-get update -y
sudo apt-get install -y git

# Clone mxnet repository. In terminal, run the commands WITHOUT "sudo"
git clone https://github.com/dmlc/mxnet.git ~/mxnet --recursive
# If building with GPU, add configurations to config.mk file:
cd ~/mxnet
cp make/config.mk .
echo "USE_CUDA=1" >>config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
echo "USE_CUDNN=1" >>config.mk
# Install MXNet for Python with all required dependencies
cd ~/mxnet/setup-utils
bash install-mxnet-ubuntu-python.sh
# We have added MXNet Python package path in your ~/.bashrc.
# Run the following command to refresh environment variables.
source ~/.bashrc
# Install MXNet Python package
#!/usr/bin/env bash

######################################################################
# This script installs MXNet for Python along with all required dependencies on a Ubuntu Machine.
# Tested on Ubuntu 16.04 + distro.
######################################################################

MXNET_HOME="$HOME/mxnet/"
echo "MXNet root folder: $MXNET_HOME"

echo "Installing build-essential, libatlas-base-dev, libopencv-dev, pip, graphviz ..."
sudo apt-get update
sudo apt-get install -y build-essential libatlas-base-dev libopencv-dev

echo "Building MXNet core. This can take few minutes..."
cd "$MXNET_HOME"
make -j$(nproc)

echo "Installing Python setuptools..."
sudo apt-get install -y python-setuptools python-pip
sudo pip install --upgrade pip

echo "Installing Numpy..."
sudo pip install numpy

echo "Installing Python package for MXNet..."
cd python; sudo python setup.py install

echo "Adding MXNet path to your ~/.bashrc file"
echo "export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc

echo "Install Graphviz for plotting MXNet network graph..."
sudo pip install graphviz

echo "Installing Jupyter notebook..."
sudo pip install jupyter

echo "Done! MXNet for Python installation is complete. Go ahead and explore MXNet with Python :-)"
Contact GitHub API Training Shop Blog About
