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
import assistive_gym
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sys.path.append('./envs/cartpole-envs')
# sys.path.append('./')
# import cartpole_envs
#import highway_env

from utils import plot_reward, plot_index
from mpc.mpc_ar import MPC
from baselines.NN import NN


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

config = load_config('config/config_assistant_robot_NN.yml')
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
env = gym.make("FeedingJaco-v0")
task = [env]
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

total_tasks = 1

"""NN pretrain"""
pretrain_episodes = 10
env = task[0]
    # data collection
for epi in range(pretrain_episodes):
    obs = env.reset()
    done = False
    mpc_controller.reset()
    while not done:
        action = env.action_space.sample()
        obs_next, reward, done, info = env.step(action)
#             print(obs_next-obs)
        model.data_process([0, obs, action, obs_next - obs])
        obs = obs_next

model.validation_flag = True
model.fit()


"""testing the model with MPC while training """
test_episode = 300
save_every = 5
log = []
eps = 1
eps_decay = 0.95

for epi in range(test_episode):
    acc_reward = 0
    obs = env.reset()
    O, A, R, acc_reward, done = [], [], [], 0, False
    mpc_controller.reset()
    while not done:
        action = env.action_space.sample()
        if random.random() > eps:
            action = mpc_controller.act(model=model, state=obs)
        obs_next, reward, done, info = env.step(action)
        A.append(action)
        O.append(obs_next)
        R.append(reward)
        # append data but not training
        model.data_process([0, obs, action, obs_next - obs])
        obs = obs_next
        acc_reward += reward
        
#         print(obs[-1])
    print('epi_num', epi, 'acc_reward: ', acc_reward, 'success_food',info['task_success'], 'spilled_food',info['spilled_food'], 'distance', np.linalg.norm(obs[6:9]), 'eps', eps)
    model.fit()
    eps *= eps_decay
    if epi != 0 and epi % save_every == save_every-1:
        if log_name is None:
            log_name = time.strftime("%Y%m%d_%H%M%S")
        torch.save(model.model.state_dict(), './log/ar_nn_{}_{}.pth'.format(log_name, epi))
        print('model saved at'+log_name)

# 
#     if done:
#         samples = {
#             "obs": np.array(O),
#             "actions": np.array(A),
#             "rewards": np.array(R), 
#             "reward_sum": acc_reward,
#         }
#         log.append(samples)
#         if log_name is None:
#             log_name = datetime.datetime.now()
#         path = './misc/log/' + log_name.strftime("%d-%H-%M") + '.npy'
#         np.save(path, log, allow_pickle=True)
#         dumb_reward_plot(path)


