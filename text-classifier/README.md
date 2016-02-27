# Text Classifier Tuning

Example using SigOpt to tune logistic regression model for text sentiment classification.

More details about this example can be found in [the associated blog post](http://blog.sigopt.com/post/133089144983/sigopt-for-ml-automatically-tuning-text).

## Setup

1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile).
3. Insert your `client_token` into sigopt_creds.py
4. `git clone https://github.com/sigopt/sigopt-examples.git`
5. `cd sigopt-examples/text-classifier/`
4. `sudo ./setup_env.sh`

## Run

```
nohup python sentiment_classifier.py &
```

You can track the progress on your [experiment dashboard](https://sigopt.com/experiment/list).
