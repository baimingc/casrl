'''
@Author:
@Email: 
@Date: 2020-04-01 01:26:48
@LastEditTime: 2020-04-24 11:59:33
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
sys.path.append('./')
import cartpole_envs
#import highway_env

from utils import plot_reward, plot_index
from mpc.mpc_cp import MPC
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

    config = load_config('config/config_test.yml')
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
    print(gym_config)

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

    total_tasks = 4

    """NN pretrain"""
    pretrain_episodes = 10
    for task_idx in range(total_tasks):
        env = task[task_idx]
        # data collection
        for epi in range(pretrain_episodes):
            obs = env.reset()
            done = False
            mpc_controller.reset()
            while not done:
                action = env.action_space.sample()
                obs_next, reward, done, state_next = env.step(action)
                model.data_process([0, obs, action, obs_next - obs])
                obs = copy.deepcopy(obs_next)

    #print('collected data: ', len(data))
    # training the model
    model.validation_flag = True
    #model.n_epochs = 20
    model.fit()

    """testing the model with MPC while training """
    test_episode = 3
    test_epoch = 20
    log = []
    for ep in range(test_epoch):
        print('epoch: ', ep)
        for task_idx in range(total_tasks):
            env = task[task_idx]
            print('task: ', task_idx)
            for epi in range(test_episode):
                #print('episode: ', epi)
                acc_reward = 0
                obs = env.reset()

                O, A, R, acc_reward, done = [], [], [], 0, False
                mpc_controller.reset()
                i = 0
                while not done:
                    i+= 1

                    # env.render()
                    env_copy = prepare_dynamics(gym_config)[task_idx]
                    env_copy.reset()
                    action = np.array([mpc_controller.act(task=env_copy, model=model, state=obs)])
                    obs_next, reward, done, state_next = env.step(action)
                    A.append(action)
                    O.append(state_next)
                    R.append(reward)

                    # append data but not training
                    model.data_process([0, obs, action, obs_next - obs])
                    obs = copy.deepcopy(obs_next)
                    acc_reward += reward
                    # logger.info('reward: {}', reward)
                    #time.sleep(0.1)
                print('task: ', task_idx,'step: ', i, 'acc_reward: ', acc_reward)
                env.close()

                if done:
                    samples = {
                        "obs": np.array(O),
                        "actions": np.array(A),
                        "rewards": np.array(R),
                        "reward_sum": acc_reward,
                    }
                    print('******************')
                    print('acc_reward', acc_reward)
                    print('******************')
                    log.append(samples)
                    if log_name is None:
                        log_name = datetime.datetime.now()
                    path = './misc/log/' + log_name.strftime("%d-%H-%M") + '.npy'
                    np.save(path, log, allow_pickle=True)
                    dumb_reward_plot(path)

            # use the collected date to train model
            print('fitting the model...')
            #model.n_epochs = 20
            model.fit()