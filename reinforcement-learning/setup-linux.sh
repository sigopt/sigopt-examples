#!/bin/bash
sudo apt-get update
sudo apt-get install -y libfreetype6-dev libxft-dev libjpeg-dev
sudo apt-get install -y libatlas-base-dev gfortran
# install pip
sudo apt-get install -y python-pip python-dev build-essential
sudo apt-get install -y python-numpy
sudo pip install --upgrade pip
sudo pip install cython
sudo pip install sigopt
sudo pip install gym
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl
