'''
@Author: Mengdi Xu
@Email: 
@Date: 2020-03-22 19:46:01
@LastEditTime: 2020-04-02 12:06:41
@Description: 
'''

import sys
sys.path.append('../envs/cartpole-envs')
import cartpole_envs

import gym
import numpy as np
import copy
import time

# 20 in total
# 'CartPoleEnvPoleM01l05-v0', 'CartPoleEnvPoleM01l07-v0'
# env_name = [
#         'CartPoleEnvPoleM04l04-v0', 'CartPoleEnvPoleM04l05-v0', 'CartPoleEnvPoleM04l06-v0', 'CartPoleEnvPoleM04l07-v0',
#         'CartPoleEnvPoleM06l04-v0', 'CartPoleEnvPoleM06l05-v0', 'CartPoleEnvPoleM06l06-v0', 'CartPoleEnvPoleM06l07-v0',
#         'CartPoleEnvPoleM07l04-v0', 'CartPoleEnvPoleM07l05-v0', 'CartPoleEnvPoleM07l06-v0', 'CartPoleEnvPoleM07l07-v0',
#         'CartPoleEnvPoleM08l04-v0', 'CartPoleEnvPoleM08l05-v0', 'CartPoleEnvPoleM08l06-v0', 'CartPoleEnvPoleM08l07-v0',
#         'CartPoleEnvPoleM09l04-v0', 'CartPoleEnvPoleM09l05-v0', 'CartPoleEnvPoleM09l06-v0', 'CartPoleEnvPoleM09l07-v0',
#         ]

env_name = ['CartPoleSwingUpEnvCm05Pm05Pl05-v0',
        # 'CartPoleEnvPoleM10l05-v0', 'CartPoleEnvPoleM10l15-v0',
        # 'CartPoleEnvPoleM20l05-v0', 'CartPoleEnvPoleM20l15-v0'
        ]
# env_list = []
# for i in range(len(env_name)):
#     env_list.append(gym.make(env_name[i]))

# transition probobility
transition_matrix = np.array([[0.8, 0.2, 0.0, 0.0],
                              [0.1, 0.8, 0.0, 0.1],
                              [0.1, 0.0, 0.8, 0.1],
                              [0.0, 0.0, 0.2, 0.8]])

# transition_preset = [0,0,0, 1,1,1, 2,2,2, 0,0,0, 3,3,3,
#                      0,0,0, 1,1,1, 2,2,2, 3,3, 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
transition_preset = [0,0,0,0,0,0,0,0,0,0]
random = False

max_n = len(transition_preset)-1
n = 0
env_index = transition_preset[n]
memory_buffer = []
transition_buffer = []
while n < max_n:
    print('n ', n)
    print('env_index ', env_index)

    transition_buffer.append(env_index)
    env = gym.make(env_name[env_index])

    count = 0
    while count < 200:
        observation = env.reset()
        for t in range(200):
            env.render()
            time.sleep(0.1)
            observation_old = copy.deepcopy(observation)
            action = env.action_space.sample() # random policy
            observation, reward, done, info = env.step(action)
            print('----------')
            print('t: ', t)
            print('action, ', action.shape)
            print('action, ', action)
            print('state, ', observation.shape)
            print('state, ', observation)
            print('reward: ', reward)
            memory_buffer.append([env_index, observation_old, action, observation-observation_old])
            count += 1
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()
    n += 1
    if random:
        env_index = np.random.choice([0,1,2,3], p=transition_matrix[env_index])
    else:
        env_index = transition_preset[n]

print('size ', len(memory_buffer))

if random:
    np.save('data_gym_cartpole_transition_random', transition_buffer)
    np.save('data_gym_cartpole_random', memory_buffer)
else:
    np.save('data_gym_cartpole_transition_preset', transition_buffer)
    np.save('data_gym_cartpole_preset', memory_buffer)
