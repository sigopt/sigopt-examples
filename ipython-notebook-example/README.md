# SigOpt IPython Notebook Example

Here we use SigOpt to optimze a simple 2D function within an [ipython notebook](http://ipython.org/notebook.html).

We create an experiment, form the suggestion feedback loop to optimize the function, then visualize the results against several other methods.

You can modify this notebook to optimize any function.

## Setup
1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile) and set it
  as the `SIGOPT_API_TOKEN` environment variable.
4. Run `sudo ./setup_env.sh`

## Run
```
ipython notebook
```
This command will automatically open up your web browser. Navigate to SigOpt_Introduction.ipynb, and select Cell -> Run All from the menu bar.
