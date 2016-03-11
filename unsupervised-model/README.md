# Unsupervised Model Tuning

Example using SigOpt to tune a combined unsupervied supervised model for OCR recognition

More details about this example can be found in [the associated blog post](LINK).

## Setup

1. Get a free SigOpt account at [https://sigopt.com/signup](https://sigopt.com/signup)
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile).
3. Insert your `client_token` into sigopt_creds.py
4. `git clone https://github.com/sigopt/sigopt-examples.git`
5. `cd sigopt-examples/text-classifier/`
4. `sudo ./setup_env.sh`

## Optimize

Once the text classifier model tuning loop is running, you can track the progress on your [experiment dashboard](https://sigopt.com/experiment/list).
