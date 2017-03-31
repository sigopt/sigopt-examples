#!/usr/bin/env bash
######################################################################
# This script installs MXNet for Python along with all required dependencies on a Amazon Linux Machine.
######################################################################
set -ex
#########
# aws ec2 run-instances --image-id ami-2757f631 --instance-type p2.8xlarge --key-name useast1 --ebs-optimized --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 32 } } ]"
# ssh ubuntu@<public-dns>
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update -y
sudo apt-get install -y cuda
#sudo apt install -y awscli
#aws configure
aws s3 cp s3://blog-nvidia-mxnet/cudnn-8.0-linux-x64-v5.1.tgz .
#cudaNN
tar xvzf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo ldconfig
##########
