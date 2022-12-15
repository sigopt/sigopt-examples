[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Using SigOpt to tune Time Series Classifiers with Keras and Tensorflow


## Summary

The data that comes with this example is sequence data extracted from diatoms. We encourage interested readers to replicate this blog using other datasets see how _optimal hyperparameter settings are data dependent_. There are over 80 different sequential datasets in the [UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/) that can be used to replicate this example. Learn more at the associated blog post: [Deep Learning Hyperparameter Optimization with Competing Objectives](https://devblogs.nvidia.com/parallelforall/sigopt-deep-learning-hyperparameter-optimization/).

In this example, SigOpt _carves out an efficient [Pareto frontier](https://en.wikipedia.org/wiki/Pareto_efficiency)_, simultaneously looking for hyperparameter configurations which yield fast inference times and high classifier accuracy.

> _Limited Access: For those interested in replicating this blog post, let us know so that we can provide a SigOpt account for you beyond your free trial!_


## EC2 Setup

This example is made for the US East (N. Virginia) region. Follow the instructions from [here](../dnn-tuning-nvidia-mxnet) to get your AWS EC2 GPU-enabled instance up and running with the following specifications:

  > - OS: Ubuntu 16.04
  > - GPU Driver: CUDA 8.0, cuDNN 5.1
  > - GPU: NVIDIA K80 GPU
  > - Server: Amazon EC2’s P2 instances
  > - DNN Library: Keras 2.0.5 on Tensorflow 1.2.0

## Instructions to Replicate Blog Post

> Post an issue or email us if you have any questions.

1. Log in to your SigOpt account at [https://app.sigopt.com](https://app.sigopt.com)

2. Copy this folder over to your EC2 instance.

  - `scp -r multimetric-timeseries/ ubuntu@<hostname>:/home/ubuntu`

3. From your EC2 instance, install SigOpt, Keras, Tensorflow and Pandas.

  - `sudo pip install sigopt tensorflow-gpu pandas`

4. Create a file named _sigopt.secret_ and the following line, entering your API key (see the [API tokens page](http://www.sigopt.com/tokens)):

  - `{"SIGOPT_API_TOKEN":"<your-api-key>"}`

5. Navigate to the folder:

  - `cd multimetric-timeseries`

7. Run the example!

  - `python main.py` (This can take a while, you may want to run it with `nohup` or `tmux`.)

8. Check out your [experiment dashboard](http://www.sigopt.com/experiments) to view your experiment progress!
