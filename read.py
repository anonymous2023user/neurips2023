import numpy as np
import scipy
from config import *
import tensorflow as tf
import csv


def load_data(path_data):

    if ext == '.mat':
        dataset = scipy.io.loadmat(path_data)
        data = dataset['dataset']
        print("Dataset has ", np.shape(data)[0], " samples.")
        lbl = dataset['label']
        x_test = data
        y_test = lbl
    elif ext == '.csv':
        file = open(path_data)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
            res = [float(i) for i in row]
            rows.append(res)
        rows = np.asarray(rows)
        print(np.shape(rows))
        y_test = rows[:, 0]
        x_test = rows[:, 1:]
    if input_shape is not None and np.shape(x_test)[1:] != input_shape:
        x_test = x_test.reshape((-1,) + input_shape)
    x_test_n = (x_test - mean) / std

    return x_test, x_test_n, y_test, mean, std


def load_net(path_net):

    modelNN = tf.keras.models.load_model(path_net)

    return modelNN


def model_properties(model):
    model.compile(optimizer='sgd', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
    W_model = model.get_weights()
    model.summary()
    n_neu = dict()
    n_neu_cum = dict()
    tmp = np.prod(list(model.layers[0].input_shape)[1:])
    n_neu[0] = [tmp]
    n_neu_cum[0] = [tmp]
    W = dict()
    layer_type = dict()
    layer_activation = dict()

    k_layer = 1
    i_weight = 0
    for k in range(len(model.layers)):
        if k == len(model.layers) - 1 and model.layers[k].__class__.__name__ == 'Dense':  # last layer
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [W_model[i_weight], W_model[i_weight + 1]]
            layer_type[k_layer] = 'Dense'
            layer_activation[k_layer] = 'none'
            n_neu[k_layer] = [tmp]
            n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]

        elif model.layers[k].__class__.__name__ == 'Conv1D':
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [W_model[i_weight], W_model[i_weight + 1], [model.layers[k].strides],
                          [model.layers[k].padding]]
            layer_type[k_layer] = 'Conv1D'
            layer_activation[k_layer] = model.layers[k].activation.__name__
            if layer_activation[k_layer] == 'None':
                n_neu[k_layer] = [tmp]
                n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]
            else:
                n_neu[k_layer] = [tmp, tmp]
                n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp,
                                      n_neu_cum[k_layer - 1][-1] + np.sum(n_neu[k_layer])]
            i_weight += 2
            k_layer += 1

        elif model.layers[k].__class__.__name__ == 'MaxPooling1D':
            if model.layers[k].pool_size[0] != model.layers[k].strides[0] or model.layers[k].padding != 'valid':
                raise Exception(
                    "Sorry, this framework only supports neural networks that has same size for pooling and striding and 'valid' padding")
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [[model.layers[k].pool_size[0]], [], [model.layers[k].strides[0]], [model.layers[k].padding]]
            layer_type[k_layer] = 'MaxPooling1D'
            layer_activation[k_layer] = 'none'
            n_neu[k_layer] = [tmp]
            n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]
            k_layer += 1

        elif model.layers[k].__class__.__name__ == 'Conv2D':
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [W_model[i_weight], W_model[i_weight + 1], [model.layers[k].strides],
                          [model.layers[k].padding]]
            layer_type[k_layer] = 'Conv2D'
            layer_activation[k_layer] = model.layers[k].activation.__name__
            if layer_activation[k_layer] == 'None':
                n_neu[k_layer] = [tmp]
                n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]
            else:
                n_neu[k_layer] = [tmp, tmp]
                n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp,
                                      n_neu_cum[k_layer - 1][-1] + np.sum(n_neu[k_layer])]
            i_weight += 2
            k_layer += 1

        elif model.layers[k].__class__.__name__ == 'MaxPooling2D':
            if model.layers[k].pool_size != model.layers[k].strides or model.layers[k].padding != 'valid':
                raise Exception(
                    "Sorry, this framework only supports neural networks that has same size for pooling and striding and 'valid' padding")
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [[model.layers[k].pool_size], [], [model.layers[k].strides], [model.layers[k].padding]]
            layer_type[k_layer] = 'MaxPooling2D'
            layer_activation[k_layer] = 'none'
            n_neu[k_layer] = [tmp]
            n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]
            k_layer += 1

        elif model.layers[k].__class__.__name__ == 'Dense':
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [W_model[i_weight], W_model[i_weight + 1]]
            layer_type[k_layer] = 'Dense'
            layer_activation[k_layer] = model.layers[k].activation.__name__
            if layer_activation[k_layer] == 'none':
                n_neu[k_layer] = [tmp]
                n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]
            else:
                n_neu[k_layer] = [tmp, tmp]
                n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp,
                                      n_neu_cum[k_layer - 1][-1] + np.sum(n_neu[k_layer])]

            i_weight += 2
            k_layer += 1

        elif model.layers[k].__class__.__name__ == 'Flatten':
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [[], []]
            layer_type[k_layer] = 'Flatten'
            layer_activation[k_layer] = 'none'
            n_neu[k_layer] = [tmp]
            n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]
            k_layer += 1

        elif model.layers[k].__class__.__name__ == 'Permute':
            tmp = np.prod(list(model.layers[k].output_shape)[1:])
            W[k_layer] = [[], []]
            layer_type[k_layer] = 'Permute'
            layer_activation[k_layer] = 'none'
            n_neu[k_layer] = [tmp]
            n_neu_cum[k_layer] = [n_neu_cum[k_layer - 1][-1] + tmp]
            k_layer += 1

        elif model.layers[k].__class__.__name__ == 'Dropout':
            continue

        else:
            raise Exception(
                "Sorry, this framework only supports Dense, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, "
                "Permute layers.")

    return W, layer_type, layer_activation, n_neu, n_neu_cum


def preprocess(layer_type, x):
    if layer_type[1] == 'Conv1D':
        data = np.expand_dims(x, axis=-1) if len(np.shape(x)[1:]) == 1 else x
    elif layer_type[1] == 'Conv2D':
        data = np.expand_dims(x, axis=-1) if len(np.shape(x)[1:]) == 2 else x
    else:
        data = x
    return data
