import numpy as np

from cnn_text.graph import build_compute_graph
from cnn_text.model import get_cnn_model
from cnn_text.evaluate import evaluate_model

def calculate_objective(assignments, data, with_architecture=False):
    """Return final value to return to SigOpt API"""
    if with_architecture:
        f1 = assignments['filter_size_1']
        f2 = assignments['filter_size_2']
        f3 = assignments['filter_size_3']
        filter_list = [f1, f2, f3]
        num_filter = assignments['num_feature_maps']
    else:
        filter_list = [3,4,5]
        num_filter = 100
    cnn = build_compute_graph(vocab_size=data.vocab_size,
                        num_embed=assignments['embed_dim'],
                        sentence_size=data.sentence_size,
                        batch_size=assignments['batch_size'],
                        dropout=assignments['dropout_rate'],
                        filter_list=filter_list,
                        num_filter=num_filter
                        )
    cnn_model = get_cnn_model(cnn=cnn,
                              batch_size=assignments['batch_size'],
                              sentence_size=data.sentence_size,
                              )
    dev_acc = evaluate_model(cnn_model=cnn_model,
                             batch_size=assignments['batch_size'],
                             max_grad_norm=assignments['max_grad_norm'],
                             learning_rate=np.exp(assignments['log_learning_rate']),
                             epoch=assignments['epochs'],
                             x_train=data.x_train,
                             y_train=data.y_train,
                             x_dev=data.x_dev,
                             y_dev=data.y_dev)
    return dev_acc
