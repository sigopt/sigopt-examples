from enum import Enum

EXACT_THRESHOLD = 50


class MetricObjectives(Enum):
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'


class MetricStrategy(Enum):
    OPTIMIZE = 'optimize'
    STORE = 'store'


class ResultsAttributes(Enum):
    EXACT = "exact"
    F1 = "f1"
    NUM_PARAMETERS = "num_parameters"
    INFERENCE_TIME = "inference_time"


def get_metrics_list():
    metrics = list()
    metrics.append(dict(name=ResultsAttributes.EXACT.value, objective="maximize", strategy="optimize", threshold=EXACT_THRESHOLD))
    metrics.append(dict(name=ResultsAttributes.NUM_PARAMETERS.value, objective="minimize", strategy="optimize"))
    metrics.append(dict(name=ResultsAttributes.F1.value, objective="maximize", strategy="store"))
    metrics.append(dict(name=ResultsAttributes.INFERENCE_TIME.value, objective="minimize", strategy="store"))
    return metrics


def get_metric_names():
    return [result_attribute.value for result_attribute in ResultsAttributes]
