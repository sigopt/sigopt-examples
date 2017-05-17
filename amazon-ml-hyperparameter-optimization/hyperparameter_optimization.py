#!/usr/bin/env python
# Amazon Machine Learning Samples
# Copyright 2015 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Amazon Software License (the "License"). You may not use
# this file except in compliance with the License. A copy of the License is
# located at
#
#     http://aws.amazon.com/asl/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or
# implied. See the License for the specific language governing permissions and
# limitations under the License.
"""
Demonstrate how to create tasks on Amazon ML to train and evaluate a model for
K-fold cross-validation. The main function of this module requires the number
of folds(kfolds).

usage: build_folds.py [--name][--debug] kfolds

example:
    python build_folds.py --name 4-fold-cv-demo 4

"""
import sys
import logging
import argparse
import config
import threading
import math
from fold import Fold
from evaluation import Evaluation
from collections import namedtuple
from hyperparameters import HYPERPARAMETERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(config.APP_NAME)
logging.getLogger('botocore.vendored.requests.packages.urllib3.connectionpool').setLevel(logging.WARN)

def build_folds(data_spec=None, kfolds=None):
    """
    Create Fold objects that will build Datasources for each fold.

    Args:
        data_spec: the named tuple object that wraps dataset related
            parameters.
        kfolds: the integer number representing the number of folds.
    Returns:
        a list of newly created Fold objects.
    """
    folds = [
        Fold(data_spec=data_spec, this_fold=i, kfolds=kfolds)
        for i
        in range(kfolds)
    ]

    for f in folds:
        f.build()  # each fold creates a Datasource
        logger.debug(f)

    return folds

def cleanup_folds(folds):
    """
    Cleanup resources that were created when building the folds

    Args:
        folds: Fold objects
    """
    for f in folds:
        f.cleanup()

def build_model_spec(regularization_type, regularization_amount):
    """
    Builds a ModelSpec for later model evaluation

    Args:
        assignmnets: a dict with keys log_regularization_amount and regularization_type
    """
    return ModelSpec(
        recipe=recipe,
        ml_model_type="BINARY",
        sgd_maxPasses="10",
        sgd_maxMLModelSizeInBytes="104857600",  # 100MiB
        sgd_RegularizationAmount=regularization_amount,
        sgd_RegularizationType=regularization_type,
    )

def build_evaluations(model_spec, folds):
    """
    Create Evaluation objects to build ML Models and Evaluations for each fold

    Args:
        model_spec: the named tuple object that wraps model related parameters
        folds: Fold objects to evaluate the model on
    Returns:
        a list of newly created Evaluation objects
    """
    evaluations = [
        Evaluation(model_spec=model_spec, fold=fold)
        for fold
        in folds
    ]

    for e in evaluations:
        e.build()  # each evaluation creates an Evaluation and an ML Model
        logger.debug(e)

    return evaluations

def cleanup_evaluations(evaluations):
    """
    Cleanup resources that were created when building the evaluations

    Args:
        evaluations: Evaluation objects
    """
    for e in evaluations:
        e.cleanup()

def collect_performance(evaluations):
    """
    Collects performance for evaluations. Spawns threads to poll and wait for each Evaluation to
        complete, then computes stats from the auc metric of each Evaluation.

    Args:
        evaluations: a list of Evaluation objects
    Returns:
        a tuple of the average of the auc metrics, and the standard deviation of the auc metrics
    """
    threads = []
    for evaluation in evaluations:
        t = threading.Thread(target=Evaluation.poll_eval, args=(evaluation,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    avg_auc = sum([e.auc for e in evaluations]) / float(len(evaluations))
    var_auc = sum([(e.auc - avg_auc) ** 2 for e in evaluations]) / float(len(evaluations))
    std_auc = math.sqrt(var_auc)

    return (avg_auc, std_auc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="%(prog)s [--name][--debug] kfolds",
        description="Demo code to create entities on Amazon ML for \
            K-fold cross-validation."
    )
    parser.add_argument(
        "kfolds",
        type=int,
        choices=range(2, 11),  # 2 to 10 is valid input
        help="the number of folds for cross-validation"
    )
    parser.add_argument(
        "-n",
        "--name",
        default="CV sample",
        help="the name of entities to create on Amazon ML"
             "[default: '%(default)s']",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="enable debug mode, logging from DEBUG level"
             "[default: off]",
        )

    args = parser.parse_args()
    if (args.debug):
        logger.setLevel(logging.DEBUG)  # modify the logging level

    logger.debug("User inputs:")
    logger.debug(vars(args))

    kfolds = args.kfolds
    name = args.name

    DataSpec = namedtuple("DataSpec", [
        "name",
        "data_s3_url",
        "schema"
    ])
    ModelSpec = namedtuple("ModelSpec", [
        "recipe",
        "ml_model_type",
        "sgd_maxPasses",
        "sgd_maxMLModelSizeInBytes",
        "sgd_RegularizationAmount",
        "sgd_RegularizationType",
    ])

    # read datasource schema and training recipe from files:
    with open("banking.csv.schema", 'r') as schema_f:
        schema = schema_f.read()
    with open("recipe.json", 'r') as recipe_f:
        recipe = recipe_f.read()

    data_spec = DataSpec(
        name=name,
        data_s3_url="s3://aml-sample-data/banking.csv",
        schema=schema,
    )

    folds = build_folds(data_spec=data_spec, kfolds=kfolds)

    best_assignments = None
    best_auc = None
    for assignments in HYPERPARAMETERS:
        model_spec = build_model_spec(assignments=assignments)
        evaluations = build_evaluations(model_spec=model_spec, folds=folds)
        (avg_auc, std_auc) = collect_performance(evaluations)
        if avg_auc > best_auc:
            best_assignments = assignments
        cleanup_evaluations(evaluations)

    cleanup_folds(folds)

    print """Best assignments are:
 - Regularization Type: {regularization_type}
 - Regularization Amount: {regularization_amount}""".format(**assignments)
