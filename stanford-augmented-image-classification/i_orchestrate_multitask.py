import sigopt
import logging
from enum import Enum


class Multitask(Enum):
    full = 1.0
    medium = 0.50
    small = 0.10


def get_assignments(*args):
    orchestrate_assignments = dict()
    for hyperparameter_enum_class in args:
        for hyperparameter_enum in hyperparameter_enum_class:
            orchestrate_assignments[hyperparameter_enum.value] = sigopt.get_parameter(
                hyperparameter_enum.value)

    logging.info("sigopt assignments being used: {}".format(orchestrate_assignments))

    task = sigopt.get_task()
    percentage_epochs = Multitask[task.name].value

    return orchestrate_assignments, percentage_epochs

