# How are My Hyperparameter Affecting My Training Time?

Example using SigOpt to explore approximated how long it takes to train your models. 

## Setup

1. Get a free SigOpt account at [https://sigopt.com/signup](https://sigopt.com/signup)
2. Find your `api_token` on your [user profile](https://sigopt.com/user/profile).
3. Recommended: Set the environment variable `SIGOPT_API_TOKEN` to your API token. Alternative: insert your api token into the jupyter notebook.
4. Clone the repo and install dependencies:

```
git clone https://github.com/sigopt/sigopt-examples.git
cd sigopt-examples/estimated-training-time/
pip install -r requirements.txt
```

## Run

We recommend using [Jupyter](http://jupyter.readthedocs.org/en/latest/install.html) to walk through this example. Run Jupyter:

```
jupyter notebook
```

Then open [`How are My Hyperparameter Affecting My Training Time?.ipynb`](https://github.com/sigopt/sigopt-examples/blob/master/estimated-training-time/How%20are%20My%20Hyperparameters%20Affecting%20My%20Training%20Time%3F.ipynb) in the browser window that pops up.

## Share
Discover something cool or interesting? Email <contact@sigopt.com> or tweet to [@SigOpt](twitter.com/sigopt) to let us know what you find! Happy Optimizing!
