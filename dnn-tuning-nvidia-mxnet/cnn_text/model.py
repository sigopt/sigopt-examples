from collections import namedtuple

import mxnet as mx

# Define the structure of our CNN Model (as a named tuple)
CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])


def get_cnn_model(cnn, batch_size, sentence_size):
    # Define what device to train/test on
    ctx = mx.gpu(0)
    # If you have no GPU on your machine change this to
    # ctx=mx.cpu(0)

    arg_names = cnn.list_arguments()

    input_shapes = {}
    input_shapes['data'] = (batch_size, sentence_size)

    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if name in ['softmax_label', 'data']: # input, output
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    param_blocks = []
    arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
    initializer = mx.initializer.Uniform(0.1)
    for i, name in enumerate(arg_names):
        if name in ['softmax_label', 'data']: # input, output
            continue
        initializer(name, arg_dict[name])

        param_blocks.append( (i, arg_dict[name], args_grad[name], name) )

    out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))

    data = cnn_exec.arg_dict['data']
    label = cnn_exec.arg_dict['softmax_label']

    cnn_model= CNNModel(cnn_exec=cnn_exec,
    					symbol=cnn, data=data,
    					label=label, param_blocks=param_blocks)

    return cnn_model
