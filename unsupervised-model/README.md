# Unsupervised Model Tuning

Example using SigOpt to tune a combined unsupervied supervised model for OCR recognition

More details about this example can be found in [the associated blog post](http://blog.sigopt.com/post/140871698423/sigopt-for-ml-unsupervised-learning-with-even).

## Setup

1. Get a free SigOpt account at [https://sigopt.com/signup](https://sigopt.com/signup)
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile) and set it
  as the `SIGOPT_API_TOKEN` environment variable.
4. `git clone https://github.com/sigopt/sigopt-examples.git`
5. `cd sigopt-examples/unsupervised-model/`
4. `sudo ./setup_env.sh`

## Optimize

Once the unsupervised model is being optimized, you can track the progress on your [experiment dashboard](https://sigopt.com/experiment/list).
