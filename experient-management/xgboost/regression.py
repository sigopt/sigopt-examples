# -*- coding: utf-8 -*-
"""Train, track, and optimize an XGBoost model.

This program trains and evaluates an XGBoost model. It can be run independently,
or inside of a `sigopt (run|optimize)` environment.

Example:
    Execute one training with the defaults and track the results in SigOpt.

        $ sigopt run xgboost.py

    Optimize the parameters used for xgboost

        $ sigopt optimize xgboost.py

sigopt.com
"""

from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import numpy as np
import sigopt
import os


def get_data():
    """Fetch data, and transform it into the format needed by xgboost and
    evaluations.

    Returns:
        dtrain: the training data in DMatrix format
        dtest: the testing (evaluation) data in DMatrix format
        ytest: the testing labels for evaluation
    """
    ds = datasets.fetch_california_housing()
    sigopt.log_dataset(" datasets.fetch_california_housing()")
    X_train, X_test, y_train, y_test = train_test_split(
        ds.data, ds.target, test_size=0.2, random_state=42
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    return (dtrain, dtest, y_test)


if __name__ == "__main__":
    """Train and evaluate an xgboost model"""

    param_defaults = {
        "learning_rate": 0.3,
        "max_depth": 5,
        "lambda": 1,
        "alpha": 0,
        "min_split_loss": 1,
        "min_child_weight": 0,
        "max_delta_step": 0,
        "subsample": 1,
        "colsample_bytree": 1,
        "colsample_bylevel": 1,
        "colsample_bynode": 1,
        "objective": "reg:squarederror",
    }

    params = dict(
        (k, sigopt.get_parameter(k, default=v)) for k, v in param_defaults.items()
    )
    num_round = sigopt.get_parameter("num_round", default=20)

    dtrain, dtest, ytest = get_data()

    sigopt.log_model("xgboost")
    evals = [(dtest, "test")]
    evals_result = {}
    bst = xgb.train(
        params,
        dtrain,
        num_round,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=False,
    )
    rmse = evals_result["test"]["rmse"][-1]

    sigopt.log_metric("rmse", rmse)
    print(f"Trained model with RMSE of {rmse}")
