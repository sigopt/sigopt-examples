[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Caffe2 Convolutional Neural Network Tuning Python Example

Example using SigOpt to tune a convolutional neural network for OCR recognition

This example is an extension of the Caffe2 MNIST example that can be found [here](https://caffe2.ai/docs/tutorial-MNIST.html).

## Setup

Note: The easiest way to try this example is to use [Amazon's Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B01M0AXXQB) as it has Caffe2 preinstalled. If you want to do this on your local machine, you can visit Caffe2's [website](https://caffe2.ai/) for Caffe2 installation instructions.

1. Get a free SigOpt account at [https://sigopt.com/signup](https://sigopt.com/signup)
2. Find your `client_token` on the [API tokens page](https://sigopt.com/tokens).
3. `git clone https://github.com/sigopt/sigopt-examples.git`
4. `cd sigopt-examples/caffe2-cnn/`
5. `sudo ./setup.sh`

## Optimize

Once the CNN model is being optimized, you can track the progress on your [experiment dashboard](https://sigopt.com/experiments).

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible.
