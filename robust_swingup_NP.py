import time, datetime
import copy
import os
import sys
import warnings

import numpy as np
from loguru import logger
import yaml
from utils import dumb_reward_plot
import gym
import torch

sys.path.append('./envs/cartpole-envs')
sys.path.append('./')
import cartpole_envs

from utils import plot_reward, plot_index
from mpc.mpc_cp import MPC
from baselines.NP_epi import NP

def prepare_dynamics(gym_config):
    dynamics_name = gym_config['dynamics_name']
    seed = gym_config['seed']
    dynamics_set = []
    for i in range(len(dynamics_name)):
        dynamics_set.append(gym.make(dynamics_name[i]))
    task = [dynamics_set[i] for i in gym_config['task_dynamics_list']]
    return task

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)
        
if __name__ == '__main__':
    # config = load_config('config/config_cpstable_np.yml')
    config = load_config('config/config_swingup_robust.yml')
    mpc_config = config['mpc_config']
    prior_safety = mpc_config['prior_safety']
    gym_config = config['gym_config']
    render = gym_config['render']
    np_config = config['NP_config']

    model = NP(NP_config=np_config)
    logger.info('Using model: {}', model.name)

    mpc_controller = MPC(mpc_config=mpc_config)

    # prepare task
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


    """NP pretrain"""
    pretrain_episodes = 3
    for task_idx in range(len(task)):
        env = task[task_idx]
        for epi in range(pretrain_episodes):
            obs = env.reset()
            done = False
            mpc_controller.reset()
            while not done:
                action = env.action_space.sample()
                obs_next, reward, done, _ = env.step(action)
                model.data_process([0, obs, action, obs_next - obs])
                obs = obs_next
        model.fit()
        model.reset()

    """testing the model with MPC while training """
    test_episode = 1
    test_epoch = 10
    log = []
    for ep in range(test_epoch):
        for task_idx in range(len(task)):
            env = task[task_idx]
            task_steps = 0
            for epi in range(test_episode):
                acc_reward = 0
                obs = env.reset()
                O, A, R, acc_reward, done, V = [], [], [], 0, False, []
                mpc_controller.reset()
                i = 0
                while not done:
                    i+= 1
                    env_copy = prepare_dynamics(gym_config)[task_idx]
                    env_copy.reset()
                    if task_steps > 0:
                        action = np.array([mpc_controller.act(task=env_copy, model=model, state=obs, ground_truth=True)])
                    else:
                        action = np.array([0.0])
                    obs_next, reward, done, violation = env.step(action)
                    task_steps += 1
                    A.append(action)
                    O.append(obs_next)
                    R.append(reward)
                    V.append(violation)

                    model.data_process([0, obs, action, obs_next - obs])
                    obs = obs_next
                    acc_reward += reward
                print('task: ', task_idx,'step: ', i, 'acc_reward: ', acc_reward, 'violation_rate: ', sum(V)/len(V))
                env.close()

                if done:
                    samples = {
                        "obs": np.array(O),
                        "actions": np.array(A),
                        "rewards": np.array(R),
                        "reward_sum": acc_reward,
                        "violation_rate": sum(V)/len(V)
                    }
                    log.append(samples)
                    if log_name is None:
                        log_name = datetime.datetime.now()
                    if not prior_safety:
                        path = './misc/log/np_robust_' + log_name.strftime("%d-%H-%M") + '.npy'
                    else:
                        path = './misc/log/np_safe_robust_' + log_name.strftime("%d-%H-%M") + '.npy'
                    np.save(path, log, allow_pickle=True)
                    dumb_reward_plot(path)
                model.fit()
            model.reset()
            
    torch.save(model.state_dict(), './misc/log/np_robust_' + log_name.strftime("%d-%H-%M") + '.pth')
    #final testing
    print('final testing')
    test_episode = 1
    test_epoch = 3
    log = []
    for ep in range(test_epoch):
        for task_idx in range(len(task)):
            env = task[task_idx]
            task_steps = 0
            for epi in range(test_episode):
                acc_reward = 0
                obs = env.reset()
                O, A, R, acc_reward, done, V = [], [], [], 0, False, []
                mpc_controller.reset()
                i = 0
                while not done:
                    i+= 1
                    env_copy = prepare_dynamics(gym_config)[task_idx]
                    env_copy.reset()
                    if task_steps > 0:
                        action = np.array([mpc_controller.act(task=env_copy, model=model, state=obs, ground_truth=True)])
                    else:
                        action = np.array([0.0])
                    obs_next, reward, done, violation = env.step(action)
                    task_steps += 1
                    A.append(action)
                    O.append(obs_next)
                    R.append(reward)
                    V.append(violation)

                    model.data_process([0, obs, action, obs_next - obs])
                    obs = obs_next
                    acc_reward += reward
                print('task: ', task_idx,'step: ', i, 'acc_reward: ', acc_reward, 'violation_rate: ', sum(V)/len(V))
                env.close()

                if done:
                    samples = {
                        "obs": np.array(O),
                        "actions": np.array(A),
                        "rewards": np.array(R),
                        "reward_sum": acc_reward,
                        "violation_rate": sum(V)/len(V)
                    }
                    log.append(samples)
                    if log_name is None:
                        log_name = datetime.datetime.now()
                    if not prior_safety:
                        path = './misc/log/np_robust_' + log_name.strftime("%d-%H-%M") + '.npy'
                    else:
                        path = './misc/log/np_safe_robust_' + log_name.strftime("%d-%H-%M") + '.npy'
                    np.save(path, log, allow_pickle=True)
                    dumb_reward_plot(path)
            model.reset()