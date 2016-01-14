# Text Classifier Tuning

Example using SigOpt to tune logistic regression model for text sentiment classification.

More details about this example can be found in [the associated blog post](http://blog.sigopt.com/post/133089144983/sigopt-for-ml-automatically-tuning-text).

Setup:

1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your SigOpt API token at https://sigopt.com/user/profile
3. insert your \<CLIENT_TOKEN\> into sigopt_creds.py
4. sudo ./setup_env.sh

Run:

nohup python sentiment_classifier.py &
