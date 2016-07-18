# Text Classifier Tuning

Example using SigOpt to tune logistic regression model for text sentiment classification.

More details about this example can be found in [the associated blog post](http://blog.sigopt.com/post/133089144983/sigopt-for-ml-automatically-tuning-text).

## Setup

1. Get a free SigOpt account at [https://sigopt.com/signup](https://sigopt.com/signup)
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile) and set it
  as the `SIGOPT_API_TOKEN` environment variable.
4. `git clone https://github.com/sigopt/sigopt-examples.git`
5. `cd sigopt-examples/text-classifier/`
4. `sudo ./setup_env.sh`

## Run

We recommend using [Jupyter](http://jupyter.readthedocs.org/en/latest/install.html) to walk through this example. Start Jupyter (run `jupyter notebook`), then open:

[`SigOpt Text Classifier Walkthrough.ipynb`](https://github.com/sigopt/sigopt-examples/blob/master/text-classifier/SigOpt%20Text%20Classifier%20Walkthrough.ipynb)

The classifier tuning can also be run without Jupyter with this command:

```
nohup python sentiment_classifier.py &
```

## Optimize

Once the text classifier model tuning loop is running, you can track the progress on your [experiment dashboard](https://sigopt.com/experiment/list).
