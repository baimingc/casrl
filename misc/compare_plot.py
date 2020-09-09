'''
@Author: 
@Email: 
@Date: 2020-05-06 13:33:02
@LastEditTime: 2020-05-06 20:21:17
@Description: 
'''

import pandas as pd
import numpy as np
from utils import plot_reward
import matplotlib.pyplot as plt


def dumb_subtask_reward_plot(path, labels, block_height=10,
                             num_subtask=4, num_subepisode=3, subtask_length=200,
                             PREFIX='', xlim=[0, 25], ylim=[0, 200], y_line1=1, y_line2=175):
    
    name = PREFIX + '_subtask_reward'
    plt.figure(figsize=(6, 4))
    plt.title(PREFIX + ' Subtask Reward')
    if y_line2:
        plt.axhline(y=y_line2, color='k')

    color_list = ['royalblue', 'lightcoral', 'c']
    for i in range(len(path)):
        acc_r_list = []

        for j in range(len(path[i])):
            inner_acc_r_list = []
            data_list = np.load(path[i][j], allow_pickle=True)
            for k, data in enumerate(data_list):
                if k > xlim[1]: 
                    break
                acc_r = data["reward_sum"]
                inner_acc_r_list.append(acc_r)
            acc_r_list.append(inner_acc_r_list)
        
        var = np.std(acc_r_list, axis=0)
        acc_r_list = np.mean(acc_r_list, axis=0)
        plt.plot(range(xlim[1]+1), acc_r_list, color_list[i], linewidth=1.5, label=labels[i])
        plt.fill_between(range(xlim[1]+1), acc_r_list-var, acc_r_list+var, facecolor=color_list[i], alpha=0.2)
    plt.grid()

    # '''
    color_bar = ['slateblue', 'deeppink', 'orange', 'slategray']
    for p_i in range(1, xlim[1]+1):
        ind = p_i % (num_subtask*num_subepisode)
        # if p_i in p_1:
        if num_subepisode*0 <= ind < num_subepisode*1:
            c_i = color_bar[0]
        # elif p_i in p_2:
        elif num_subepisode*1 <= ind < num_subepisode*2:
            c_i = color_bar[1]
        # elif p_i in p_3:
        elif num_subepisode*2 <= ind < num_subepisode*3:
            c_i = color_bar[2]
        else:
            c_i = color_bar[3]
        plt.fill_between([p_i-1, p_i], -block_height, ylim[0], facecolor=c_i, alpha=0.5)
    # '''

    plt.ylim([ylim[0]-block_height, ylim[1]])
    plt.xlim(xlim)
    plt.ylabel('subtask accumulated reward')
    plt.xlabel('Number of data points/' + str(subtask_length))
    plt.legend(loc=4)
    plt.savefig('./misc/' + name, dpi=200)
    plt.close('all')
    #plt.show()


directory = './misc/log/'
labels = ['DPGP', 'GP', 'NN']

"""For cartpole"""
path = [
            [
                directory + 'CartPole-DPGP-15-19-13.npy', 
                directory + 'CartPole-DPGP-06-02-01.npy', 
                directory + 'CartPole-DPGP-05-20-37.npy',  
            ],
            [
                directory + 'CartPole-SingleGP-06-08-01.npy', 
                directory + 'CartPole-SingleGP-06-12-21.npy'
            ],
            [
                directory + 'CartPole-NN-06-11-32.npy',
                directory + 'CartPole-NN-06-11-19.npy'
            ]
        ]
dumb_subtask_reward_plot(path, labels=labels,
                         num_subtask=4,
                         PREFIX='CartPole',
                         xlim=[0, 26],
                         ylim=[0, 200], y_line1=1, y_line2=175)


"""For Intersection"""
path = [
            [
                directory + 'Intersection-DPGP-27-22-21.npy',
                directory + 'Intersection-DPGP-06-15-54.npy',
                directory + 'Intersection-DPGP-06-18-32.npy'
            ],
            [
                directory + 'Intersection-SingleGP-06-09-30.npy',
                directory + 'Intersection-SingleGP-06-13-19.npy'
            ],
            [
                directory + 'Intersection-NN-08-15-54.npy',
                directory + 'Intersection-NN-08-15-55.npy'
            ]
        ]
dumb_subtask_reward_plot(path, labels=labels, PREFIX='Intersection', block_height=7,
                         num_subtask=3, subtask_length=40,
                         xlim=[0, 25],
                         ylim=[0, 140], y_line1=1, y_line2=107)
