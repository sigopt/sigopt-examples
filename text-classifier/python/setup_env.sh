#!/bin/bash
OS_NAME="$(uname -s)"
if [ "$OS_NAME" == 'Darwin' ]; then
  brew install gfortran
  sudo easy_install pip
else
  # assuming using ubuntu
  sudo apt-get update
  sudo apt-get -y install python-pip python-dev libopenblas-dev liblapack-dev gfortran
fi
sudo pip install -r requirements.txt
curl https://public.sigopt.com/NEGATIVE_list.json -O
curl https://public.sigopt.com/POSITIVE_list.json -O
