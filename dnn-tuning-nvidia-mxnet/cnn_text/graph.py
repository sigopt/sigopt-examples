import sys

import mxnet as mx

def build_compute_graph(vocab_size, num_embed, sentence_size,
                        batch_size, dropout, filter_list, num_filter):
    '''
    Define place holders for network inputs and outputs
    '''

    input_x = mx.sym.Variable('data') # placeholder for input data
    input_y = mx.sym.Variable('softmax_label') # placeholder for output label

    '''
    Define the first network layer (embedding)
    '''
    embed_layer = mx.sym.Embedding(data=input_x,
                                   input_dim=vocab_size,
                                   output_dim=num_embed,
                                   name='vocab_embed')

    # reshape embedded data for next layer
    conv_input = mx.sym.Reshape(data=embed_layer,
                                target_shape=(batch_size, 1,
                                              sentence_size,
                                              num_embed))

    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input,
                                   kernel=(filter_size, num_embed),
                                   num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui,
                               pool_type='max',
                               kernel=(sentence_size - filter_size + 1, 1),
                               stride=(1,1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)

    # reshape for next layer
    h_pool = mx.sym.Reshape(data=concat,
                            target_shape=(batch_size, total_filters))

    # dropout layer
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

    # fully connected layer
    num_label=2

    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    fc = mx.sym.FullyConnected(data=h_drop,
                               weight=cls_weight,
                               bias=cls_bias,
                               num_hidden=num_label)

    # softmax output
    sm = mx.sym.SoftmaxOutput(data=fc,
                              label=input_y,
                              name='softmax')

    # set CNN pointer to the "back" of the network
    cnn = sm

    return cnn
