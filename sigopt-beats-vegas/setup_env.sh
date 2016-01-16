#!/bin/bash
OS_NAME="$(uname -s)"
if [ '$OS_NAME' == 'Darwin' ]; then
  brew install gfortran
  sudo easy_install pip
else
  # assuming using ubuntu
  sudo apt-get update
  sudo apt-get -y install libffi-dev python-pip python-dev libssl-dev libxml2-dev libxslt1-dev python-scipy gfortran
fi
sudo pip install -r requirements.txt
cd boxscores/scraper
./scrape_all
cd ../..
