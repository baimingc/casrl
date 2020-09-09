'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 10:41:36
@LastEditTime: 2020-03-25 00:27:20
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

#training_envs = ['PendulumEnv_070-070-v0']
training_env_params = [
            (0.7, 0.7),
            (0.9, 0.9),
            (0.7, 0.9),
            (0.9, 0.7)
        ]

test_env_params = [
            (0.8, 0.8),
            (1.0, 1.0),
            (0.8, 1.0),
            (1.0, 0.8)
        ]

training_envs, test_envs = get_env_names(
    "PendulumEnv", training_env_params, test_env_params)

seed = 1
envs_train = initialize_envs(training_envs, seed)
envs_test = initialize_envs(test_envs, seed)

env = envs_train[0]

tqdm.write("************* Begin warm up *************")
state_action_pairs, delta_states = warm_up(env, episodes=100, max_step=200, render=False)


dynamic_model = DynamicModel(config)

controller = MPC(env, config, dynamic_model)

tqdm.write("Fitting the model")
loss = dynamic_model.fit(state_action_pairs, delta_states)

tqdm.write("[INFO] loss train: %.4f, loss test:  %.4f" % (loss[0], loss[1]))

test_reward = []
state = env.reset()
for i in range(150):
    action = controller.act(state)
    #print(action)
    state, reward, done, _ = env.step(np.array([action]))
    test_reward.append(reward)
    env.render()

tqdm.write("Test reward: %4f"%np.mean(test_reward))

tqdm.write("************* Begin MPC iteration.... *************")
mpc_itr = 100

#t = trange(mpc_itr)
t = tqdm(range(mpc_itr), position=0)
t.set_description(f"mpc itr: [{0}/{mpc_itr}]")
for itr in t: #range(mpc_itr):
    state_action_pairs_mpc, delta_states_mpc = controller.mpc_itr(episodes=50,
                                                                  max_step=150,
                                                                  render=False)

    #samples_left = int(np.floor(len(state_action_pairs)*1))
    #state_action_pairs = state_action_pairs[:samples_left]
    #delta_states = delta_states[:samples_left]
    state_action_pairs += state_action_pairs_mpc
    delta_states += delta_states_mpc
    loss = dynamic_model.fit(state_action_pairs, delta_states)

    test_reward = []
    state = env.reset()
    for i in range(150):
        action = controller.act(state)
        state, reward, done, _ = env.step(np.array([action]))
        test_reward.append(reward)
        env.render()
    tqdm.write("[INFO] mpc itr: [%i/%i], test mean reward: %.2f, loss train: %.4f, loss test:  %.4f"
               % (itr, mpc_itr, np.mean(test_reward), loss[0], loss[1]))
    t.set_description(f"mpc itr: [{itr}/{mpc_itr}]")
    #t.update(1)
