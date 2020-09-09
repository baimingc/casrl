'''
@Author:
@Email: 
@Date: 2020-04-01 01:26:48
@LastEditTime: 2020-04-13 10:33:04
@Description: 
'''

import time, datetime
import copy
import os
import sys
import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from loguru import logger
import yaml
from utils import dumb_reward_plot

import gym
sys.path.append('./envs/cartpole-envs')
sys.path.append('./envs/highway-env')
import cartpole_envs
#import highway_env

from utils import plot_reward, plot_index
from mpc.mpc_cp import MPC

# all models
from dpgpmm.DPGPMM import DPGPMM
# from baselines.SingleGP import SingleGP
# from baselines.SingleSparseGP import SingleSparseGP
from baselines.NN import NN


def prepare_dynamics(gym_config):
    dynamics_name = gym_config['dynamics_name']
    seed = gym_config['seed']
    dynamics_set = []
    for i in range(len(dynamics_name)):
        env = gym.make(dynamics_name[i])
        # env.seed(seed)
        dynamics_set.append(gym.make(dynamics_name[i]))
    
    # use pre-defined env sequence
    task = [dynamics_set[i] for i in gym_config['task_dynamics_list']]
    return task


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)


if __name__ == '__main__':
    # dynamic model configuration
    # config = load_config('config_DPGP_MBRL.yml')
    config = load_config('config_MBRL.yml')
    dpgp_config = config['DPGP_config']
    gp_config = config['SingleGP_config']
    sparse_gp_config = config['SingleSparseGP_config']
    nn_config = config['NN_config']
    mpc_config = config['mpc_config']
    gym_config = config['gym_config']
    render = gym_config['render']

    # initialize the mixture model
    # model = DPGPMM(dpgp_config=dpgp_config)
    # model = SingleSparseGP(sparse_gp_config=sparse_gp_config)
    # model = SingleGP(gp_config=gp_config)
    model = NN(NN_config=nn_config)
    logger.info('Using model: {}', model.name)

    # initial MPC controller
    mpc_controller = MPC(mpc_config=mpc_config)

    # prepare task
    # the task is solved, if each dynamic is solved
    task = prepare_dynamics(gym_config)

    """start DPGP-MBRL"""
    data_buffer = []
    label_list = []
    subtask_list = []
    subtask_reward = []
    subtask_succ_count = [0]
    comp_trainable = [1]
    task_reward = []
    trainable = True
    task_solved = False
    subtask_solved = [False, False, False, False]
    total_count = 0
    task_epi = 0

    log_name = None
    log, A, O, R = [], [], [], []

    """NN pretrain"""
    env = task[0]
    env_copy = prepare_dynamics(gym_config)[0]
    env_copy.reset()
    # data collection
    pretrain_episodes = 20
    for epi in range(pretrain_episodes):
        acc_reward = 0
        obs, state = env.reset()
        O.append(state)

        done = False
        mpc_controller.reset()
        i = 0
        while not done:
            i+= 1
            env.render()
            print('step : ', i)
            action = np.array([mpc_controller.act(task=env_copy, model=model, state=state, ground_truth=True)])
            state_next, reward, done, info = env.step(action)
            state = copy.deepcopy(state_next)
            print('action ', action)
            print('reward: %.4f' % reward)

            acc_reward += reward
            A.append(action)
            O.append(state_next)
            R.append(reward)
        env.close()
        samples = {
            "obs": np.array(O),
            "actions": np.array(A),
            "rewards": np.array(R),
            "reward_sum": acc_reward,
        }
        log.append(samples)
        if log_name is None:
            log_name = datetime.datetime.now()
        path = './misc/log/GT_' + log_name.strftime("%d-%H-%M") + '.npy'
        np.save(path, log, allow_pickle=True)
        dumb_reward_plot(path, PREFIX='GT_')