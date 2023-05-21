import numpy as np
from gurobipy import Model, GRB, quicksum
from astropy.nddata import reshape_as_blocks
from config import *


def init_pert(center, delta, mean, std):

    in_shape = np.shape(center)
    lower = np.reshape(center - delta, in_shape + (1,))
    upper = np.reshape(center + delta, in_shape + (1,))
    if type(mean) == list and len(np.shape(lower)) == 4:
        lower = (lower - np.reshape(mean, (1, 1, len(mean), 1))) / np.reshape(std, (1, 1, len(std), 1))
        upper = (upper - np.reshape(mean, (1, 1, len(mean), 1))) / np.reshape(std, (1, 1, len(std), 1))
    elif type(mean) == list and len(np.shape(lower)) == 3:
        lower = (lower - np.reshape(mean, (1, len(mean), 1))) / np.reshape(std, (1, len(std), 1))
        upper = (upper - np.reshape(mean, (1, len(mean), 1))) / np.reshape(std, (1, len(std), 1))
    elif type(mean) == list and len(np.shape(lower)) == 2:
        lower = (lower - np.reshape(mean, (len(mean), 1))) / np.reshape(std, (len(std), 1))
        upper = (upper - np.reshape(mean, (len(mean), 1))) / np.reshape(std, (len(std), 1))
    else:
        lower = (lower - mean) / std
        upper = (upper - mean) / std

    return lower, upper


def bound_prop(weight, bias, lower_pre, upper_pre, operator_type, act_func):

    if operator_type == 'Dense' or operator_type == 'Conv1D' or operator_type == 'Conv2D':
        n_neu = np.shape(weight)[-1]
        if act_func == 'none':
            num_cols = 1
        else:
            num_cols = 2
        ll = np.zeros([n_neu, num_cols])
        uu = np.zeros([n_neu, num_cols])

    if operator_type == 'Dense':
        W_pos = np.maximum(weight[:, :], 0)
        W_neg = np.minimum(weight[:, :], 0)
        lower_exp = np.expand_dims(lower_pre[:, -1], axis=1)
        upper_exp = np.expand_dims(upper_pre[:, -1], axis=1)
        ll[:, 0] = np.sum(W_pos * lower_exp + W_neg * upper_exp, axis=0) + bias[:]
        uu[:, 0] = np.sum(W_pos * upper_exp + W_neg * lower_exp, axis=0) + bias[:]
        if act_func == 'relu':
            ll[:, 1] = np.maximum(0, ll[:, 0])
            uu[:, 1] = np.maximum(0, uu[:, 0])

    elif operator_type == 'Conv1D':
        W_pos = np.maximum(weight[:, :, :], 0)
        W_neg = np.minimum(weight[:, :, :], 0)
        lower_exp = np.expand_dims(lower_pre[:, :, -1], axis=2)
        upper_exp = np.expand_dims(upper_pre[:, :, -1], axis=2)
        ll[:, 0] = np.sum(W_pos * lower_exp + W_neg * upper_exp, axis=(0, 1)) + bias[:]
        uu[:, 0] = np.sum(W_pos * upper_exp + W_neg * lower_exp, axis=(0, 1)) + bias[:]
        if act_func == 'relu':
            ll[:, 1] = np.maximum(0, ll[:, 0])
            uu[:, 1] = np.maximum(0, uu[:, 0])

    elif operator_type == 'MaxPooling1D':
        pool_size = weight[0]
        stride = bias[0][0]
        padding = bias[1][0]
        if pool_size == stride and padding == 'valid':
            last_neuron = pool_size * int(np.shape(lower_pre)[0] / pool_size)
            num_filter = np.shape(lower_pre)[1]
            ll = np.expand_dims(np.amax(np.reshape(lower_pre[:last_neuron, :, 1], [-1, pool_size, num_filter]), axis=1),
                                axis=-1)
            uu = np.expand_dims(np.amax(np.reshape(upper_pre[:last_neuron, :, 1], [-1, pool_size, num_filter]), axis=1),
                                axis=-1)

    elif operator_type == 'Conv2D':
        W_pos = np.maximum(weight[:, :, :, :], 0)
        W_neg = np.minimum(weight[:, :, :, :], 0)
        lower_exp = np.expand_dims(lower_pre[:, :, :, -1], axis=3)
        upper_exp = np.expand_dims(upper_pre[:, :, :, -1], axis=3)
        ll[:, 0] = np.sum(W_pos * lower_exp + W_neg * upper_exp, axis=(0, 1, 2)) + bias[:]
        uu[:, 0] = np.sum(W_pos * upper_exp + W_neg * lower_exp, axis=(0, 1, 2)) + bias[:]
        if act_func == 'relu':
            ll[:, 1] = np.maximum(0, ll[:, 0])
            uu[:, 1] = np.maximum(0, uu[:, 0])

    elif operator_type == 'MaxPooling2D':
        pool1, pool2 = weight[0]
        stride1, stride2 = bias[0][0]
        padding = bias[1][0]
        last_neuron1 = stride1 * int((np.shape(lower_pre)[0] - pool1) / stride1) + pool1
        last_neuron2 = stride2 * int((np.shape(lower_pre)[1] - pool2) / stride2) + pool2
        num_filter = np.shape(lower_pre)[2]
        if pool1 == stride1 and pool2 == stride2 and padding == 'valid':
            ll = np.amax(reshape_as_blocks(lower_pre[:last_neuron1, :last_neuron2, :, 1], (pool1, pool2, num_filter)),
                         axis=(-2, -3))
            uu = np.amax(reshape_as_blocks(upper_pre[:last_neuron1, :last_neuron2, :, 1], (pool1, pool2, num_filter)),
                         axis=(-2, -3))
            if len(np.shape(ll)) == 4 and np.shape(ll)[2] == 1:
                ll = np.swapaxes(ll, 2, 3)
                uu = np.swapaxes(uu, 2, 3)

    elif operator_type == 'Flatten':
        if len(np.shape(lower_pre)) == 3:
            ll = lower_pre[:, :, -1].reshape((-1, 1))
            uu = upper_pre[:, :, -1].reshape((-1, 1))
        elif len(np.shape(lower_pre)) == 4:
            ll = lower_pre[:, :, :, -1].reshape((-1, 1))
            uu = upper_pre[:, :, :, -1].reshape((-1, 1))

    elif operator_type == 'Permute':
        if len(np.shape(lower_pre)) == 3:
            ll = np.expand_dims(lower_pre[:, :, -1].transpose((1, 0)), axis=-1)
            uu = np.expand_dims(upper_pre[:, :, -1].transpose((1, 0)), axis=-1)
        elif len(np.shape(lower_pre)) == 4:
            ll = np.expand_dims(lower_pre[:, :, :, -1].transpose((2, 0, 1)), axis=-1)
            uu = np.expand_dims(upper_pre[:, :, :, -1].transpose((2, 0, 1)), axis=-1)

    return ll, uu


def get_status(lower, upper, layer_type, layer_activation):

    if layer_activation == 'relu':
        if layer_type == 'Dense':
            oas = np.asarray((np.sign(lower[:, 0]) + np.sign(upper[:, 0])) / 2, int)
            oas[np.where(lower[:, 0] == 0)] = 1
            oas[np.where(upper[:, 0] == 0)] = -1
        elif layer_type == 'Conv1D':
            oas = np.asarray((np.sign(lower[:, :, 0]) + np.sign(upper[:, :, 0])) / 2, int)
            oas[np.where(lower[:, :, 0] == 0)] = 1
            oas[np.where(upper[:, :, 0] == 0)] = -1
        elif layer_type == 'Conv2D':
            oas = np.asarray((np.sign(lower[:, :, :, 0]) + np.sign(upper[:, :, :, 0])) / 2, int)
            oas[np.where(lower[:, :, :, 0] == 0)] = 1
            oas[np.where(upper[:, :, :, 0] == 0)] = -1
    elif layer_activation == 'none':
        oas = []

    return oas


def net_propagate(k_start, W, layer_type, layer_activation, lower, upper, oas=dict(), cum=None):

    n_layers = len(layer_type)
    for i in range(k_start, n_layers + 1):
        if layer_type[i] == 'Conv1D':
            stride = W[i][2][0][0]
            output_conv = int(np.floor((np.shape(lower[i - 1])[0] - np.shape(W[i][0])[0]) / stride) + 1)
            step_conv = np.shape(W[i][0])[0]
            ll = []
            uu = []
            for it in range(0, stride * output_conv, stride):
                lower_conv_tmp, upper_conv_tmp = bound_prop(W[i][0], W[i][1], lower[i - 1][it:it + step_conv],
                                                            upper[i - 1][it:it + step_conv], layer_type[i],
                                                            layer_activation[i])
                ll.append(lower_conv_tmp)
                uu.append(upper_conv_tmp)
            ll = np.asarray(ll)
            uu = np.asarray(uu)
        elif layer_type[i] == 'Conv2D':
            stride1, stride2 = W[i][2][0]
            output_conv_1 = int(np.floor((np.shape(lower[i - 1])[0] - np.shape(W[i][0])[0]) / stride1) + 1)
            output_conv_2 = int(np.floor((np.shape(lower[i - 1])[1] - np.shape(W[i][0])[1]) / stride2) + 1)
            step_conv1, step_conv2 = np.shape(W[i][0])[0], np.shape(W[i][0])[1]
            num_filter = np.shape(W[i][0])[3]
            if layer_activation[i] == 'none':
                ll = np.zeros((output_conv_1, output_conv_2, num_filter, 1))
                uu = np.zeros((output_conv_1, output_conv_2, num_filter, 1))
            else:
                ll = np.zeros((output_conv_1, output_conv_2, num_filter, 2))
                uu = np.zeros((output_conv_1, output_conv_2, num_filter, 2))
            for it1 in range(0, stride1 * output_conv_1, stride1):
                for it2 in range(0, stride2 * output_conv_2, stride2):
                    lower_conv_tmp, upper_conv_tmp = bound_prop(W[i][0], W[i][1], lower[i - 1][it1:it1 + step_conv1,
                                                                                  it2:it2 + step_conv2],
                                                                upper[i - 1][it1:it1 + step_conv1,
                                                                it2:it2 + step_conv2], layer_type[i],
                                                                layer_activation[i])
                    ll[int(it1 / stride1), int(it2 / stride2), :, :] = lower_conv_tmp
                    uu[int(it1 / stride1), int(it2 / stride2), :, :] = upper_conv_tmp
        elif layer_type[i] == 'Dense' or layer_type[i] == 'Permute':
            ll, uu = bound_prop(W[i][0], W[i][1], lower[i - 1], upper[i - 1], layer_type[i], layer_activation[i])
        elif layer_type[i] == 'Flatten':
            ll, uu = bound_prop(W[i][0], W[i][1], lower[i - 1], upper[i - 1], layer_type[i], layer_type[i - 1])
        elif layer_type[i] == 'MaxPooling1D' or layer_type[i] == 'MaxPooling2D':
            ll, uu = bound_prop(W[i][0], (W[i][2], W[i][3]), lower[i - 1], upper[i - 1], layer_type[i],
                                layer_activation[i])

        ll = np.asarray(ll)
        uu = np.asarray(uu)
        lower[i] = ll
        upper[i] = uu
        if i != n_layers + 1:
            oas[i] = get_status(lower[i], upper[i], layer_type[i], layer_activation[i])
    if k_start == 1:
        gb_inds = dict()
        for i in range(0, n_layers + 1):
            size_ind = np.size(lower[i])
            shape_ind = np.shape(lower[i])
            if i == 0:
                shift_ind = 0
            else:
                shift_ind = cum[i - 1][-1]
            gb_inds[i] = np.arange(size_ind).reshape(shape_ind, order='F') + shift_ind

    if k_start == 1:
        return lower, upper, oas, gb_inds
    else:
        return lower, upper, oas


def model_generator(W, lower, upper, layer_type, layer_activation, n_neu, gb_inds, k_save=[]):

    gb_model = dict()
    cnstr_status = dict()
    n_layers = len(lower)
    n_neurons = np.sum([np.sum(n_neu[k]) for k in range(n_layers)])
    model = Model()
    variables = model.addVars(int(n_neurons), lb=-1 * float('inf'), name="variables")
    model.Params.LogToConsole = 0
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    for k in range(n_layers):
        if k == 0:
            inds = np.squeeze(np.reshape(gb_inds[k], (-1, 1)))
            shape_ind = np.shape(inds)
            low_tmp = lower[k].reshape(shape_ind, order='F')
            up_tmp = upper[k].reshape(shape_ind, order='F')
            for jj in range(n_neu[k][0]):
                model.addConstr(variables[inds[jj]] >= low_tmp[inds[jj]])
                model.addConstr(variables[inds[jj]] <= up_tmp[inds[jj]])
            cnstr_status[k] = 1
        elif 0 < k < n_layers:
            if layer_type[k] == 'Conv1D':
                stride = W[k][2][0][0]
                output_shape = np.shape(lower[k])[0]
                step_conv = np.shape(W[k][0])[0]
                num_filter = np.shape(lower[k])[1]
                for f in range(num_filter):
                    ind_list_all = gb_inds[k - 1][:, :, -1]
                    for m in range(0, output_shape * stride, stride):
                        ind_m = gb_inds[k][m, f, 0]
                        ind_list = np.squeeze(ind_list_all[m:m + step_conv].reshape((-1, 1), order='F'))
                        W_f = np.squeeze(W[k][0][:, :, f].reshape((-1, 1), order='F'))
                        model.addConstr(quicksum(
                            W_f[z] * variables[ind_list[z]] for z in
                            range(np.size(ind_list))) - variables[ind_m] == -1 * W[k][1][f])
                if layer_activation[k] == 'none':
                    cnstr_status[k] = 1
                else:
                    cnstr_status[k] = 0
            elif layer_type[k] == 'MaxPooling1D':
                cnstr_status[k] = 0
            elif layer_type[k] == 'Conv2D':
                stride1, stride2 = W[k][2][0]
                (output_shape1, output_shape2) = np.shape(lower[k])[:2]
                step_conv1, step_conv2 = np.shape(W[k][0])[:2]
                num_filter = np.shape(W[k][0])[3]
                for f in range(num_filter):
                    ind_list_all = gb_inds[k - 1][:, :, :, -1]
                    for m1 in range(0, output_shape1 * stride1, stride1):
                        for m2 in range(0, output_shape2 * stride2, stride2):
                            ind_m = gb_inds[k][int(m1 / stride1), int(m2 / stride2), f, 0]
                            ind_list = np.squeeze(
                                ind_list_all[m1:m1 + step_conv1, m2:m2 + step_conv2].reshape((-1, 1), order='F'))
                            W_f = np.squeeze(W[k][0][:, :, :, f].reshape((-1, 1), order='F'))
                            model.addConstr(quicksum(
                                W_f[z] * variables[ind_list[z]] for z in
                                range(np.size(ind_list))) - variables[ind_m] == -1 * W[k][1][f])
                if layer_activation[k] == 'none':
                    cnstr_status[k] = 1
                else:
                    cnstr_status[k] = 0
            elif layer_type[k] == 'MaxPooling2D':
                cnstr_status[k] = 0
            elif layer_type[k] == 'Permute':
                inds_k = np.squeeze(gb_inds[k][:, :, :, -1].reshape((-1, 1), order='F'))
                inds = np.squeeze(gb_inds[k - 1][:, :, :, -1].transpose(2, 0, 1).reshape((-1, 1), order='F'))
                for m in range(n_neu[k][0]):
                    model.addConstr(
                        variables[inds_k[m]] == variables[inds[m]])
                cnstr_status[k] = 1
            elif layer_type[k] == 'Flatten':
                if len(np.shape(gb_inds[k - 1])) == 3:
                    inds = np.squeeze(gb_inds[k - 1][:, :, -1].reshape((-1, 1)))  # , order='F'))
                elif len(np.shape(gb_inds[k - 1])) == 4:
                    inds = np.squeeze(gb_inds[k - 1][:, :, :, -1].reshape((-1, 1)))  # , order='F'))
                for m in range(n_neu[k][0]):
                    model.addConstr(
                        variables[gb_inds[k][m, -1]] == variables[inds[m]])
                cnstr_status[k] = 1
            elif layer_type[k] == 'Dense':
                for m in range(n_neu[k][0]):
                    ind_m = gb_inds[k][m, 0]
                    model.addConstr(quicksum(
                        W[k][0][z, m] * variables[gb_inds[k - 1][z, -1]] for z in
                        range(n_neu[k - 1][-1])) - variables[ind_m] == -1 * W[k][1][m])
                if layer_activation[k] == 'none':
                    cnstr_status[k] = 1
                else:
                    cnstr_status[k] = 0

        if k in k_save:
            model.update()
            gb_model[k] = model.copy()

    return gb_model, cnstr_status


def create_model_gb(k_start, k_end, model_gb, W, layer_type, layer_activation, lower, upper, oas, gb_inds, cnstr_status,
                    n_neu, n_neu_cum, k_save=[]):

    model = model_gb.copy()
    variables = model.getVars()
    for k in range(k_start, k_end):
        if cnstr_status[k] == 0:
            if layer_type[k] == 'Conv1D':
                output_shape = np.shape(lower[k])[0]
                num_filter = np.shape(lower[k])[1]
                for f in range(num_filter):
                    for m in range(output_shape):
                        ind_m = gb_inds[k][m, f, 0]
                        if layer_activation[k] == 'relu':
                            ind_j = gb_inds[k][m, f, 1]
                            if oas[k][m, f] == 1:
                                model.addConstr(variables[ind_j] == variables[ind_m])
                            elif oas[k][m, f] == -1:
                                model.addConstr(variables[ind_j] == 0)
                            elif oas[k][m, f] == 0:
                                model.addConstr(variables[ind_j] >= 0)
                                model.addConstr(variables[ind_j] - variables[ind_m] >= 0)
                                model.addConstr(
                                    variables[ind_j] - upper[k][m, f, 0] * (variables[ind_m] - lower[k][m, f, 0]) /
                                    (upper[k][m, f, 0] - lower[k][m, f, 0]) <= 0)
            elif layer_type[k] == 'MaxPooling1D':
                pool_size = W[k][0][0]
                last_neu = pool_size * int(np.shape(lower[k - 1])[0] / pool_size)
                lower_fit = lower[k - 1][:last_neu, :, 1]
                upper_fit = upper[k - 1][:last_neu, :, 1]
                shape_fit = np.shape(lower_fit)[1]
                num_filter = np.shape(lower[k])[1]
                low_max_all_filts = np.repeat(np.amax(np.reshape(lower_fit[:, :], [-1, pool_size, shape_fit]), axis=1),
                                              pool_size, axis=0)
                states = 1 - (low_max_all_filts >= upper_fit)
                ind_all = gb_inds[k - 1][:, :, -1]
                count_pools = 0
                for f in range(num_filter):
                    for m in range(0, last_neu, pool_size):
                        ind_p = n_neu_cum[k - 1][-1] + count_pools
                        if not states[m:m + pool_size, f].any(axis=0):  # lowers and uppers are equal
                            model.addConstr(variables[ind_p] == variables[
                                ind_all[m + np.argmax(upper_fit[m:m + pool_size, f]), f]])
                        else:
                            if np.sum(states[m:m + pool_size, f]) == 1:  # One lower is larger than all other uppers
                                model.addConstr(variables[ind_p] == variables[
                                    ind_all[m + np.where(states[m:m + pool_size, f] == 1)[0][0], f]])
                            else:  # includes over-approximation
                                model.addConstr(quicksum(
                                    states[m + z, f] * variables[ind_all[m + z, f]] for z in range(pool_size)) -
                                                variables[ind_p] >= np.sum(
                                    states[m:m + pool_size, f] * lower_fit[m:m + pool_size, f]) - np.max(
                                    states[m:m + pool_size, f] * lower_fit[m:m + pool_size, f]))
                                for i in range(pool_size):
                                    if states[m + i, f] != 0:
                                        model.addConstr(variables[ind_p] >= variables[ind_all[m + i, f]])
                        count_pools += 1
            elif layer_type[k] == 'Conv2D':
                output_shape1, output_shape2 = np.shape(lower[k])[:2]
                num_filter = np.shape(lower[k])[2]
                for f in range(num_filter):
                    for m1 in range(output_shape1):
                        for m2 in range(output_shape2):
                            ind_m = gb_inds[k][m1, m2, f, 0]
                            if layer_activation[k] == 'relu':
                                ind_j = gb_inds[k][m1, m2, f, 1]
                                if oas[k][m1, m2, f] == 1:
                                    model.addConstr(variables[ind_j] == variables[ind_m])
                                elif oas[k][m1, m2, f] == -1:
                                    model.addConstr(variables[ind_j] == 0)
                                elif oas[k][m1, m2, f] == 0:
                                    model.addConstr(variables[ind_j] >= 0)
                                    model.addConstr(variables[ind_j] - variables[ind_m] >= 0)
                                    model.addConstr(
                                        variables[ind_j] - upper[k][m1, m2, f, 0] * (
                                                variables[ind_m] - lower[k][m1, m2, f, 0]) /
                                        (upper[k][m1, m2, f, 0] - lower[k][m1, m2, f, 0]) <= 0)
            elif layer_type[k] == 'MaxPooling2D':
                pool1, pool2 = W[k][0][0]
                stride1, stride2 = W[k][2][0]
                last_neuron1 = stride1 * int((np.shape(lower[k - 1])[0] - pool1) / stride1) + pool1
                last_neuron2 = stride2 * int((np.shape(lower[k - 1])[1] - pool2) / stride2) + pool2
                num_filter = np.shape(lower[k])[2]
                lower_fit = lower[k - 1][:last_neuron1, :last_neuron2, :, 1]
                upper_fit = upper[k - 1][:last_neuron1, :last_neuron2, :, 1]
                low_max_all_filts = np.amax(reshape_as_blocks(lower_fit[:, :, :], (pool1, pool2, num_filter)),
                                            axis=(-2, -3))
                if len(np.shape(low_max_all_filts)) == 4 and np.shape(low_max_all_filts)[2] == 1:
                    low_max_all_filts = np.squeeze(low_max_all_filts, 2)
                low_max_all_filts = low_max_all_filts.repeat(pool1, axis=0).repeat(pool2, axis=1)
                states = 1 - (low_max_all_filts >= upper_fit)
                ind_all = gb_inds[k - 1][:, :, :, -1]
                count_pools = 0
                for f in range(num_filter):
                    for m1 in range(0, last_neuron1, stride1):
                        for m2 in range(0, last_neuron2, stride2):
                            ind_p = gb_inds[k][int(m1 / stride1), int(m2 / stride2), f, 0]
                            if not states[m1:m1 + pool1, m2:m2 + pool2, f].any(
                                    axis=(0, 1)):  # lowers and uppers are equal
                                a = np.argmax(upper_fit[m1:m1 + pool1, m2:m2 + pool2, f])
                                model.addConstr(variables[ind_p] == variables[
                                    ind_all[m1 + int(a / pool1), m2 + a % pool1, f]])
                            else:
                                if np.sum(states[m1:m1 + pool1, m2:m2 + pool2, f]) == 1:
                                    a, b = np.where(states[m1:m1 + pool1, m2:m2 + pool2,
                                                    f] == 1)  # One lower is larger than all other uppers
                                    model.addConstr(variables[ind_p] == variables[
                                        ind_all[m1 + a[0], m2 + b[0], f]])
                                else:  # includes over-approximation
                                    model.addConstr(quicksum(
                                        states[m1 + z1, m2 + z2, f] * variables[ind_all[m1 + z1, m2 + z2, f]] for z1 in
                                        range(pool1) for z2 in range(pool2)) - variables[ind_p] >= np.sum(
                                        states[m1:m1 + pool1, m2:m2 + pool2, f] * lower_fit[m1:m1 + pool1,
                                                                                  m2:m2 + pool2, f]) - np.max(
                                        states[m1:m1 + pool1, m2:m2 + pool2, f] * lower_fit[m1:m1 + pool1,
                                                                                  m2:m2 + pool2, f]))
                                    for i1 in range(pool1):
                                        for i2 in range(pool2):
                                            if states[m1 + i1, m2 + i2, f] != 0:
                                                model.addConstr(
                                                    variables[ind_p] >= variables[ind_all[m1 + i1, m2 + i2, f]])
                            count_pools += 1
            elif layer_type[k] == 'Dense':
                for m in range(n_neu[k][0]):
                    ind_m = gb_inds[k][m, 0]
                    if layer_activation[k] == 'relu':
                        ind_j = gb_inds[k][m, 1]
                        if oas[k][m] == 1:
                            model.addConstr(variables[ind_j] == variables[ind_m])
                        elif oas[k][m] == -1:
                            model.addConstr(variables[ind_j] == 0)
                        elif oas[k][m] == 0:
                            model.addConstr(variables[ind_j] >= 0)
                            model.addConstr(variables[ind_j] - variables[ind_m] >= 0)
                            model.addConstr(
                                variables[ind_j] - upper[k][m, 0] * (variables[ind_m] - lower[k][m, 0]) /
                                (upper[k][m, 0] - lower[k][m, 0]) <= 0)
        if k in k_save:
            model.update()
            model_gb_new = model.copy()
    model.update()
    if len(k_save) == 0:
        return model
    else:
        return model_gb_new, model


def check_verifciation(model_ver, num_classes, lbl, gb_inds, low_ver, num_ver):

    model = model_ver.copy()
    variables = model.getVars()
    k_last_layer = len(gb_inds) - 1
    low_list = []
    for c in range(num_classes):
        if c != int(lbl) and c not in low_ver:
            ind_class = np.squeeze(gb_inds[k_last_layer][lbl])
            ind_check = np.squeeze(gb_inds[k_last_layer][c])
            model.setObjective(variables[ind_class] - variables[ind_check], GRB.MINIMIZE)
            model.optimize()
            low_value = model.ObjVal
            low_list.append(low_value)
            if low_value < 0:
                break
            else:
                low_ver[c] = low_value
    if all(item > 0 for item in low_list):
        num_ver += 1
        status = "Verified"
    else:
        status = "Not Verified"

    return status, num_ver, low_ver


def bound_refinement(k_start, k_end, gb_model, W, layer_type, layer_activation, lower, upper, oas, gb_inds,
                     cnstr_status, n_neu, n_neu_cum):

    model = gb_model.copy()
    model = create_model_gb(k_start, k_end, model, W, layer_type, layer_activation,
                            lower, upper, oas, gb_inds, cnstr_status, n_neu, n_neu_cum)
    variables = model.getVars()
    low_list = []
    up_list = []
    ll = np.copy(lower[k_end])
    uu = np.copy(upper[k_end])
    if layer_type[k_end] == 'Dense':
        indices = np.where(oas[k_end] == 0)
        for it in range(len(indices[0])):
            ind0 = gb_inds[k_end][indices[0][it], 0]
            model.setObjective(variables[ind0], GRB.MINIMIZE)
            model.optimize()
            low = model.ObjVal
            low_list.append(low)
            model.reset()
            model.setObjective(variables[ind0], GRB.MAXIMIZE)
            model.optimize()
            up = model.ObjVal
            up_list.append(up)
            model.reset()
        if layer_activation[k_end] == 'relu':
            ll[indices[0], 0] = low_list
            ll[indices[0], 1] = np.maximum(low_list, 0)
            uu[indices[0], 0] = up_list
            uu[indices[0], 1] = np.maximum(up_list, 0)
    elif layer_type[k_end] == 'Conv1D':
        indices = np.where(oas[k_end] == 0)
        for it in range(len(indices[0])):
            ind0 = gb_inds[k_end][indices[0][it], indices[1][it], 0]
            model.setObjective(variables[ind0], GRB.MINIMIZE)
            model.optimize()
            low = model.ObjVal
            low_list.append(low)
            model.reset()
            model.setObjective(variables[ind0], GRB.MAXIMIZE)
            model.optimize()
            up = model.ObjVal
            up_list.append(up)
            model.reset()
        if layer_activation[k_end] == 'relu':
            ll[indices[0], indices[1], 0] = low_list
            ll[indices[0], indices[1], 1] = np.maximum(low_list, 0)
            uu[indices[0], indices[1], 0] = up_list
            uu[indices[0], indices[1], 1] = np.maximum(up_list, 0)
    elif layer_type[k_end] == 'Conv2D':
        indices = np.where(oas[k_end] == 0)
        for it in range(len(indices[0])):
            ind0 = gb_inds[k_end][indices[0][it], indices[1][it], indices[2][it], 0]
            model.setObjective(variables[ind0], GRB.MINIMIZE)
            model.optimize()
            low = model.ObjVal
            low_list.append(low)
            model.reset()
            model.setObjective(variables[ind0], GRB.MAXIMIZE)
            model.optimize()
            up = model.ObjVal
            up_list.append(up)
            model.reset()
        if layer_activation[k_end] == 'relu':
            ll[indices[0], indices[1], indices[2], 0] = low_list
            ll[indices[0], indices[1], indices[2], 1] = np.maximum(low_list, 0)
            uu[indices[0], indices[1], indices[2], 0] = up_list
            uu[indices[0], indices[1], indices[2], 1] = np.maximum(up_list, 0)

    return ll, uu


def save_inds(layer_activation):

    act_inds = [k for k in range(1, len(layer_activation) + 1) if layer_activation[k] == 'relu']
    k_save = [act_inds[x] - 1 for x in range(len(act_inds))]
    k_save.pop(0)
    k_save.append(len(layer_activation))

    return act_inds, k_save