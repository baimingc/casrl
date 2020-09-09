'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 01:01:24
@LastEditTime: 2020-03-24 11:07:39
@Description:
'''

import numpy as np
from mpc import MPC
from utils import *
from models.nn import DynamicModel
from tqdm import trange, tqdm

config_path = "config.yml"
config = load_config(config_path)
#print_config(config_path)
config["model_config"]["load_model"]=True

training_envs = ['PendulumEnv_070-070-v0']

seed = 1
envs = initialize_envs(training_envs, seed)

env = envs[0]

dynamic_model = DynamicModel(config)

controller = MPC(env, config, dynamic_model)

print("MPC iteration....")


for k in range(100):
    test_reward = []
    state = env.reset()
    env.render()
    for i in range(100):
        action = controller.act(state)
        #print(action)
        state, reward, done, _ = env.step(np.array([action]))
        test_reward.append(reward)
        env.render()

    print("test reward: ", np.mean(test_reward))
