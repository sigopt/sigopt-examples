#!/bin/bash

set -e

if ! [ -d data ]; then
  mkdir -p data
  cd data
  for DATA_URL in \
    https://sigopt-public.s3-us-west-2.amazonaws.com/experiment-templates/mnist/train-images \
    https://sigopt-public.s3-us-west-2.amazonaws.com/experiment-templates/mnist/train-labels \
    https://sigopt-public.s3-us-west-2.amazonaws.com/experiment-templates/mnist/test-images \
    https://sigopt-public.s3-us-west-2.amazonaws.com/experiment-templates/mnist/test-labels
  do
    wget "$DATA_URL"
  done
fi
