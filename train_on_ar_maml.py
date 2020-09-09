import time
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from loguru import logger
import yaml
from utils import dumb_reward_plot, dumb_predict_error_plot
import datetime

import gym
from baselines.MAML import MAML

import assistive_gym

import torch

from mpc.mpc_ar import MPC
from baselines.NP_epi import NP

CONFIG_NAME = './config/config_ar.yml'

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)


if __name__ == '__main__':
    # dynamic model configuration
    config = load_config(CONFIG_NAME)    
    maml_config = config['maml_config']
    mpc_config = config['mpc_config']
    gym_config = config['gym_config']
    render = gym_config['render']

    # initialize the mixture model
    model = MAML(maml_config=maml_config)
    logger.info('Using model: {}', model.name)
#     logger.info('Using environment: {}', ENVS)

    # initial MPC controller
    mpc_controller = MPC(mpc_config=mpc_config)

    # prepare meta and adaptiive task
    env = gym.make("FeedingJacoHuman-v0")
    log_name = None

    """ Meta Learning Stage """
    pretrain_episodes = 10
    for task_idx in range(1):
        a1 = np.random.uniform(0, 0.02)
        a2 = np.random.uniform(-0.005, 0)
        for epi in range(pretrain_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                state_next, reward, done, info = env.step(action)
                action[-4] = a1
                action[-1] = a2
                obs_next, reward, done, _ = env.step(action)
                model.data_process([0, obs, action[:7], obs_next - obs], used_for_adaption=False)
                obs = obs_next
            model.finish_one_episode()
    model.fit()

    """ Adapt Stage (also train meta model) """
    acc_R = []
    save_every = 5
    for ep in range(300):
        time_start = time.time()
        a1 = np.random.uniform(0, 0.1)
        a2 = np.random.uniform(-0.02, 0)
        human_action = np.array([a1, 0, 0, a2])
        for epi in range(1): # each subtask contains a fixed number of episode
            # train stage
            O, A, R, acc_reward, done, E = [], [], [], 0, False, []
            state = env.reset()
            acc_reward = 0
            
            mpc_controller.reset()
            while not done:
                robot_action = mpc_controller.act(model=model, state=state)
                action = np.concatenate((robot_action, human_action))
                state_next, reward, done, info = env.step(action)
                # adapt model
                model.data_process([0, state, action[:7], state_next-state], used_for_adaption=True)
                model.adapt()

                # interact with env

#                 state_delta, mse_error = model.test(state[None], action[None], state_next)

                # print('reward: ', reward)
                # print('prediction: ', state_delta)
                # print('groundtruth: ', state_next-state)
                # print('mse_error: ', mse_error)

                acc_reward += reward
                A.append(action)
                O.append(state_next)
                R.append(reward)

                state = state_next
            model.finish_one_episode()
            model.fit()
            print(ep, 'acc_reward: ', acc_reward, 'success_food',info['task_success'], 'spilled_food',info['spilled_food'], 'distance', np.linalg.norm(state[6:9]))
            acc_R.append(acc_reward)
            if ep % save_every == save_every -1:
                if log_name is None:
                    log_name = time.strftime("%Y%m%d_%H%M%S")
#                 torch.save(model.meta_model.state_dict(), './log/ar_maml_{}_{}.pth'.format(log_name, ep))
                torch.save(acc_R, './log/ar_maml_{}_reward.pth'.format(log_name))
                print('model saved at'+log_name)
        time_end = time.time()
        print('time used:', time_end - time_start)