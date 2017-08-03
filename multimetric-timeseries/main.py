from multiprocessing import Process, Queue

from sigopt import Connection

from config import (SIGOPT_API_TOKEN, PARAMETERS, EXPERIMENT_NAME,
                     METRICS, OBSERVATION_BUDGET, DATASET_PATH)

from train import prepare_data, evaluate_assignments

conn = Connection(client_token=SIGOPT_API_TOKEN)

experiment = conn.experiments().create(
    name=EXPERIMENT_NAME,
    parameters=PARAMETERS,
    metrics = METRICS,
    observation_budget = OBSERVATION_BUDGET
    )

nb_classes, x_train, Y_train, x_test, Y_test = prepare_data(DATASET_PATH)
q = Queue()

while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()

    p = Process(target=evaluate_assignments, args=(
        q, experiment, suggestion, x_train, Y_train,
        x_test, Y_test, nb_classes)
    )
    p.start()
    p.join()
    metrics, metadata = q.get()

    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        values = metrics,
        metadata=metadata
    )

    experiment = conn.experiments(experiment.id).fetch()
