#!/bin/bash

set -e

if ! [ -d data ]; then
  mkdir -p data
  cd data
  for DATA_URL in \
    https://s3-us-west-2.amazonaws.com/sigopt-public/experiment-templates/mnist/train-images \
    https://s3-us-west-2.amazonaws.com/sigopt-public/experiment-templates/mnist/train-labels \
    https://s3-us-west-2.amazonaws.com/sigopt-public/experiment-templates/mnist/test-images \
    https://s3-us-west-2.amazonaws.com/sigopt-public/experiment-templates/mnist/test-labels
  do
    wget "$DATA_URL"
  done
fi
