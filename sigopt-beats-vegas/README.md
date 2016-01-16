# sigopt-beats-vegas

Learn more at the associated [blog post](http://blog.sigopt.com/post/136340340198/sigopt-for-ml-using-model-tuning-to-beat-vegas).

Dive right in with [the iPython Notebook](https://github.com/sigopt/sigopt-examples/blob/master/sigopt-beats-vegas/SigOpt%20NBA%20OverUnder%20Model.ipynb).

## Setup

1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile).
3. Insert your `client_token` into sigopt_creds.py
4. Run `sudo ./setup_env.sh`

## Run

To open up the [ipython notebook](http://ipython.org/notebook.html):
```
ipython notebook
```
This command will automatically open up your web browser. Navigate to SigOpt_Introduction.ipynb, and select Cell -> Run All from the menu bar.

To run the predictor as a standalone:
```
cd predictor
python stand_alone.py
```
