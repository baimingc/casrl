import time, datetime
import copy
import os
import sys
import warnings
import torch
import numpy as np
from loguru import logger
import yaml
from utils import dumb_reward_plot
import gym
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import assistive_gym

import torch


from utils import plot_reward, plot_index
from mpc.mpc_ar import MPC
from baselines.SingleGP import SingleGP

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

config = load_config('config/config_ar.yml')
mpc_config = config['mpc_config']
gym_config = config['gym_config']
render = gym_config['render']
gp_config = config['SingleGP_config']

model = SingleGP(gp_config=gp_config)

logger.info('Using model: {}', model.name)

mpc_controller = MPC(mpc_config=mpc_config)

env = gym.make("FeedingJacoHuman-v0")
log_name = None



"""testing the model with MPC while training """
test_episode = 1
test_epoch = 300
save_every = 1
log = []
acc_R = []
for ep in range(test_epoch):
    task_steps = 0
    a1 = np.random.uniform(0, 0.1)
    a2 = np.random.uniform(-0.02, 0)
    human_action = np.array([a1, 0, 0, a2])
    for epi in range(test_episode):
        acc_reward = 0
        obs = env.reset()
        O, A, R, acc_reward, done, V = [], [], [], 0, False, []
        mpc_controller.reset()
        while not done:
            robot_action = mpc_controller.act(model=model, state=obs)
            action = np.concatenate((robot_action, human_action))
            obs_next, reward, done, info = env.step(action)
            task_steps += 1
            A.append(action)
            O.append(obs_next)
            R.append(reward)
            model.fit(data=[0, obs, action[:7], obs_next - obs])
            obs = obs_next
            acc_reward += reward
    #             print('task: ', task_idx,'step: ', i, 'acc_reward: ', acc_reward, 'violation_rate: ', sum(V)/len(V))
        print(ep, 'acc_reward: ', acc_reward, 'success_food',info['task_success'], 'spilled_food',info['spilled_food'], 'distance', np.linalg.norm(obs[6:9]))
        acc_R.append(acc_reward)
        if ep % save_every == save_every -1:
            if log_name is None:
                log_name = time.strftime("%Y%m%d_%H%M%S")
            torch.save(model.model.state_dict(), './log/ar_gp_{}_{}.pth'.format(log_name, ep))
            torch.save(acc_R, './log/ar_gp_{}_reward.pth'.format(log_name))
            print('model saved at'+log_name)