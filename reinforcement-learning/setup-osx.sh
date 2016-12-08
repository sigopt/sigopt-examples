#!/bin/bash
curl https://bootstrap.pypa.io/ez_setup.py -o - | python
sudo easy_install pip
sudo pip install --upgrade pip
sudo pip install -r requirements.txt
