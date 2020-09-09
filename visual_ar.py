import time, datetime
import os
import sys
import warnings

import numpy as np
import yaml
import gym
import assistive_gym

from mpc.mpc_ar import MPC
from baselines.NN import NN


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

config = load_config('config/config_assistant_robot_NN_load.yml')
nn_config = config['NN_config']
mpc_config = config['mpc_config']
gym_config = config['gym_config']
render = gym_config['render']

# initialize the mixture model
# model = DPGPMM(dpgp_config=dpgp_config)
# model = SingleSparseGP(sparse_gp_config=sparse_gp_config)
# model = SingleGP(gp_config=gp_config)
model = NN(NN_config=nn_config)
# logger.info('Using model: {}', model.name)

# initial MPC controller
mpc_controller = MPC(mpc_config=mpc_config)

# prepare task
# the task is solved, if each dynamic is solved
env = gym.make("FeedingJaco-v0")


"""testing the model with MPC while training """
test_episode = 3
for epi in range(test_episode):
    acc_reward = 0
#     env.render()
    obs = env.reset()
    O, A, R, acc_reward, done = [], [], [], 0, False
    mpc_controller.reset()
    while not done:
#         env.render()
        action = mpc_controller.act(model=model, state=obs)
#         print(action)
        obs_next, reward, done, info = env.step(action)
        
#         print(obs_next)
        A.append(action)
        O.append(obs_next)
        R.append(reward)
        # append data but not training
        obs = obs_next
        acc_reward += reward
    print('acc_reward: ', acc_reward, 'success_food',info['task_success'], 'spilled_food',info['spilled_food'], 'distance', np.linalg.norm(obs[6:9]))

