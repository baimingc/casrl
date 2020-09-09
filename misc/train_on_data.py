'''
@Author: Mengdi Xu, Wenhao Ding
@Email: 
@Date: 2020-03-17 12:32:22
@LastEditTime: 2020-04-02 12:22:15
@Description: 
'''

import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from loguru import logger

from utils import plot_index_results, print_three_models, plot_gibbs_time

from dpgp.DPMixture import DPMixture
#from GPComponent_pyro import GPComponent
#from GPComponent_tf import GPComponent
#from GPComponent_pytorch import GPComponent
from dpgp.GPComponent_pytorch_cartpole import GPComponent
#from GPComponent_pytorch_cartpole_sparse import GPComponent


def data_process(use_data, data, d_i, discret):
    if use_data == 'TOY':
        data_point = data[d_i].reshape((-1, 4))
        label = data_point[:, 0]
    elif use_data == 'CARTPOLE':
        # [env_index, s, a, s_next]
        s = data[d_i][1]
        if discret:
            a = np.array([data[d_i][2]])
        else:
            a = data[d_i][2]
        s_n = data[d_i][3]
        data_point = np.concatenate((s, a, s_n), axis=0)[None]
        label = data[d_i][0]
    
    return data_point, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DPGP-MBRL')
    parser.add_argument('-alpha', default=0.5, type=float, help='alpha initialization parameter')
    parser.add_argument('-ada_alpha', default=False, type=bool, help='adaptively update alpha')
    
    parser.add_argument('-use_data', default='CARTPOLE', type=str, help='selected dataset')

    parser.add_argument('-merge', default=True, type=bool, help='use merge strategy in sequential_vi or not')
    parser.add_argument('-merge_threshold', default=20.0, type=float, help='merge a component when the kld is below this value')
    parser.add_argument('-merge_burnin', default=20, type=int, help='the sample number to start merge')

    parser.add_argument('-state_dim', default=4, type=int, help='dimension of the state space')
    parser.add_argument('-action_dim', default=1, type=int, help='dimension of the action space')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate of the gaussian process')
    parser.add_argument('-gp_iter', default=3, type=int, help='iteration time of GP')

    args = parser.parse_args()
    use_data = args.use_data

    if use_data == 'TOY':
        # synthesis toy dataset
        data = np.load('./data/data_sparse.npy')
        label = None
    elif use_data == 'CARTPOLE':
        # gym cart-pole dataset
        data = np.load('./data/data_gym_cartpole_preset.npy', allow_pickle=True)
        label = np.load('./data/data_gym_cartpole_transition_preset.npy', allow_pickle=True)

    hyperparams = [2, 2, 2, 2, 2, 2, 2, 2]
    DP_mix = DPMixture(GPComponent, hyperparams, args)
    data = data[150:500]

    label_list = []
    predict_list = []
    time_record = []
    for d_i in range(data.shape[0]):
        # add one data at a time
        data_point, label = data_process(use_data, data, d_i, discret=False)
        DP_mix.add_point(data_point)

        start_time = time.time()
        # alpha = DP_mix.gibbs_sample(n_iter=1)
        # alpha = DP_mix.sequential_vi()
        alpha = DP_mix.sequential_vi_w_transition()
        time_record.append(time.time() - start_time)

        if use_data == 'TOY':
            print_three_models(DP_mix.assigns)
            label_list.append(label)
        elif use_data == 'CARTPOLE':
            logger.info("id: {}, alpha: {}", d_i, alpha)
            print("label: {}, predict: {}".format(label, DP_mix.assigns[d_i]))
            label_list.append(label)

    # record the updated assignments
    predict_list = []
    for i in range(data.shape[0]):
        predict_list.append(DP_mix.assigns[i])
    np.save('./misc/Gibbs_full_time.npy', time_record)
    plot_index_results(data, predict_list, label_list)
    plot_gibbs_time(data, time_record)
