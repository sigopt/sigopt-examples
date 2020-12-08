#!/bin/bash

set -e

if ! [ -d data ]; then
  mkdir -p data
  cd data
  for DATA_URL in \
    https://public.sigopt.com/experiment-templates/mnist/train-images \
    https://public.sigopt.com/experiment-templates/mnist/train-labels \
    https://public.sigopt.com/experiment-templates/mnist/test-images \
    https://public.sigopt.com/experiment-templates/mnist/test-labels
  do
    wget "$DATA_URL"
  done
fi
