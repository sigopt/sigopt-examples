#!/bin/bash
sudo apt-get update
sudo apt-get install -y libfreetype6-dev libxft-dev libjpeg-dev
sudo apt-get install -y libatlas-base-dev gfortran
# install pip
sudo apt-get install -y python-pip python-dev build-essential
sudo apt-get install -y python-scipy
sudo pip install --upgrade pip
sudo pip install cython
sudo pip install -U sklearn 
sudo pip install scikit-image 
sudo pip install sigopt
sudo pip install xgboost

# download SVHN dataset
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
