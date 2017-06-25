[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Using SigOpt to tune Time Series Classifiers with Keras and Tensorflow


## Summary

There are dozens of different types of time series data available at the [UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/). The data that comes with this example is sequence data extracted from diatoms. We encourage users of this to train other time series classiers on different datasets and see how **optimal hyperparameter settings are data dependent**. Learn more at the associated blog post: _TBD_.

In this example, SigOpt _carves out an effecient [Pareto frontier](https://en.wikipedia.org/wiki/Pareto_efficiency)_, simultaneously looking for hyperparameter configurations which yield fast inference times and high classifier accuracy.

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

1. [Sign up](http://sigopt.com/signup) for a SigOpt account.

2. Copy this folder over to your EC2 instance.

  - `scp -r multimetric-timeseries/ ubuntu@<hostname>:/home/ubuntu`

3. From your EC2 instance, install SigOpt, Keras, Tensorflow and Pandas.

  - `sudo pip install sigopt tensorflow-gpu keras pandas`

4. Create a file named _sigopt.secret_ and the following line, entering your API key (see your [profile](http://www.sigopt.com/user/profile)):

  - `{"SIGOPT_API_TOKEN":"<your-api-key>"}`

5. Navigate to the folder:

  - `cd multimetric-timeseries`

6. Edit _config.py_:

  - `vi config.py`

7. Run the example!

  - `nohup python main.py &`

8. Check out your [experiment dashboard](http://www.sigopt.com/experiments) to view your experiment progress!
