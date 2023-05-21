from pytictoc import TicToc
from read import *
from help_gb_funcs import *
import time

if __name__ == '__main__':
    print(path_data)
    print(path_net)
    NNmodel = load_net(path_net)
    data, data_n, labels, mean, std = load_data(path_data)
    W, layer_type, layer_activation, n_neu, n_neu_cum = model_properties(NNmodel)
    data_n = preprocess(layer_type, data_n)
    data = preprocess(layer_type, data)
    score = NNmodel.evaluate(data_n, labels, verbose=0)
    print("The accuracy of this neural network model is ", score[1] * 100, "%")
    predictions = np.squeeze(NNmodel.predict(data_n))
    print(labels)
    print(predictions)
    nums = np.shape(labels)[0]
    cases = np.ones((nums, len(deltas)))
    for i_d, delta in enumerate(deltas):
        print("Perturbation is ", delta)
        num_ver = 0
        numTrue = 0
        num_ver_healthy = 0
        num_healthy = 0
        num_time_out = 0
        print("total number of healthy patients is ", nums - np.sum(labels))
        process_time_tot = 0
        t_all = TicToc()
        t = TicToc()
        t_all.tic()  # Start timer all
        for i in range(nums):
            if cases[i, i_d] == 0:
                continue
            t.tic()  # Start timer
            timeout_start = time.time()
            current_time = 0
            if len(np.shape(labels)) == 1:
                lbl = labels[i].copy()
            else:
                lbl = labels[i][0].copy()
            if lbl == np.argmax(predictions[i]):
                print('Number is ', i + 1, ' label is ', int(lbl), ' prediction is ', np.argmax(predictions[i]))
                numTrue += 1
                num_classes = np.shape((list(W.items())[-1][1][1]))[0]
                lower = dict()
                upper = dict()
                low_ver = dict()  # lowers of last layer
                center = data[i]
                lower[0], upper[0] = init_pert(center, delta, mean=mean, std=std)
                act_inds, k_save = save_inds(layer_activation)
                lower, upper, oas, gb_inds = net_propagate(1, W, layer_type, layer_activation, lower, upper,
                                                           cum=n_neu_cum)
                gb_model, cnstr_status = model_generator(W, lower, upper, layer_type, layer_activation, n_neu, gb_inds,
                                                         k_save=k_save)
                ind_last = list(gb_model.keys())[-1]
                gb_model[ind_last], model_ver = create_model_gb(0, len(lower), gb_model[ind_last], W, layer_type,
                                                                layer_activation, lower, upper, oas, gb_inds,
                                                                cnstr_status, n_neu, n_neu_cum, k_save=[k_save[0]])
                ver_status, num_ver, low_ver = check_verifciation(model_ver, num_classes, int(lbl), gb_inds,
                                                                  low_ver, num_ver)
                for j, k in enumerate(k_save):
                    if j == 0:
                        continue
                    if ver_status != 'Verified':
                        ll, uu = bound_refinement(0, act_inds[j], gb_model[k_save[j]], W, layer_type, layer_activation,
                                                  lower, upper, oas, gb_inds, cnstr_status, n_neu, n_neu_cum)
                        lower[act_inds[j]], upper[act_inds[j]] = ll, uu
                        oas[act_inds[j]] = get_status(lower[act_inds[j]], upper[act_inds[j]], layer_type[act_inds[j]],
                                                      layer_activation[act_inds[j]])
                        lower, upper, oas = net_propagate(act_inds[j] + 1, W, layer_type, layer_activation, lower,
                                                          upper, oas)
                        gb_model[ind_last], model_ver = create_model_gb(act_inds[j], len(lower),
                                                                        gb_model[ind_last], W,
                                                                        layer_type, layer_activation, lower,
                                                                        upper, oas, gb_inds, cnstr_status,
                                                                        n_neu, n_neu_cum, k_save=[k_save[j]])
                        ver_status, num_ver_c, low_ver = check_verifciation(model_ver, num_classes, int(lbl),
                                                                                gb_inds, low_ver, num_ver)
                        current_time = time.time()
                        if current_time <= timeout_start + timeout:
                            num_ver = num_ver_c
                        else:
                            break
                if current_time <= timeout_start + timeout:
                    processing_time = t.tocvalue()
                    process_time_tot += processing_time
                    print('Number is ', i + 1, ' Accurately Classified is ', numTrue, ' Verified is ', num_ver,
                          ' Time out is ', num_time_out, ' Processing time is ', processing_time)
                else:
                    num_time_out += 1
                    print('Number is ', i + 1, ' Accurately Classified is ', numTrue, ' Verified is ', num_ver,
                          ' Time out is ', num_time_out)
                if ver_status != 'Verified':
                    cases[i, i_d:] = 0
            else:
                print('Number is ', i + 1, ' Accurately Classified is ', numTrue, ' Verified is ', num_ver,
                      ' Time out is ', num_time_out)
        print('Number is ', i + 1, ' Accurately Classified is ', numTrue, ' Verified is ', num_ver,
              ' Time out is ', num_time_out, ' Processing time is ', process_time_tot)