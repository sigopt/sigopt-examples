[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Surprise Lib (scikit-surprise) Recommendation System Python Jupyter Notebook Model Tuning

This example uses SigOpt to tune a Surprise Lib recommender model based on the SVD approach that influence the winners of the Netflix prize in the 2009-20012 time frame.

## Jupyter Setup

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/surprise-py-recsys`
3. Run `jupyter lab` in that directory and open surprise_recommender.ipynb in the web interface
4. Add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) in the Jupyter cell where you see `YOUR_API_TOKEN_HERE`
5. Run all cells or step through the notebook

## Python Setup

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/surprise-py-recsys`
3. Add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) on line 127 where you see `YOUR_API_TOKEN_HERE`
4. Run `python surprise_recommender.py` or open the file in your favorite text editor to see how it works 

## Optimize

Once the SigOpt optimization loop is initiated, you can track the progress on your [experiment dashboard](https://sigopt.com/experiments).

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more!
