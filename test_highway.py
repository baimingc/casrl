import pandas as pd
import numpy as np
from utils import plot_reward
import matplotlib.pyplot as plt

import gym
import highway_env
from train_on_swingup import load_config

# from utils import dumb_reward_plot
# path = './misc/log/IntersectionLL_23-23-29.npy'
# dumb_reward_plot(path, PREFIX='intersection_single',
#                  xlim=[0, 40], ylim=[20, 130], y_line1=3, y_line2=120)


config = load_config('./config/config_roundabout.yml')
# TODO for roundabout, change road.py line 268
env_config = config['roundabout_config']

# env = gym.make('roundabout-v1')
env = gym.make('intersectionMultiVehicle-v20')
# env = gym.make('parking-v0')
env.reset()
# env.change_config(env_config)
print('configure space: ', env.config)

for kk in range(10):
    print('kk ', kk)
    done = False
    env.reset()
    while not done:
        env.render()
        action = env.action_space.sample()
        action[0] = 0
        action[1] = 0
        print('action, ', action)
        obs, reward, done, info = env.step(action)
        # action: int
        # ACTIONS = {0: 'LANE_LEFT',
        #            1: 'IDLE',
        #            2: 'LANE_RIGHT',
        #            3: 'FASTER',
        #            4: 'SLOWER'}
        # in abstract.py
        # reward: collision, sped and lane change
        # perception distance
        # ego vehicle: MDPVehicle in control.py

        # print('----------')
        # print('duration, ', env.config["duration"])
        # print('steps: ', env.steps)
        # print('action, ', action)
        # print('state, ', obs)
        # print('reward: ', reward)
        # print('info: ', info)
    env.close()
