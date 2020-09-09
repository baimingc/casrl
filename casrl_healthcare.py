import time, datetime
import os
import numpy as np
import yaml
import gym
import assistive_gym
import torch

from mpc.mpc_hc import MPC
from model.NP import NP

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

config = load_config('config/config_healthcare.yml')
mpc_config = config['mpc_config']
np_config = config['NP_config']

model = NP(NP_config=np_config)
mpc_controller = MPC(mpc_config=mpc_config)
env = gym.make("FeedingJacoHuman-v0")

"""testing the model with MPC while training """
test_episode = 200
save_every = 5
acc_R = []
log_name = None
model.reset()

for ep in range(test_episode):
    task_steps = 0
    ## sample patient action
    a1 = np.random.uniform(-0.1, 0.1)
    a2 = np.random.uniform(-0.2, 0.2)
    human_action = np.array([a1, 0, 0, a2])

    obs = env.reset()
    O, A, R, acc_reward, done = [], [], [], 0, False
    mpc_controller.reset()
    while not done:
        if task_steps > 0:
            robot_action = mpc_controller.act(model=model, state=obs)
        else:
            robot_action = np.zeros(7)
        action = np.concatenate((robot_action, human_action))
        obs_next, reward, done, info = env.step(action)
        task_steps += 1
        A.append(action)
        O.append(obs_next)
        R.append(reward)
        model.data_process([0, obs, action[:7], obs_next - obs])
        obs = obs_next
        acc_reward += reward
    print(ep, 'acc_reward: ', acc_reward, 'success_food',info['task_success'], 'spilled_food',info['spilled_food'], 'distance', np.linalg.norm(obs[6:9]))
    acc_R.append(acc_reward)
    if ep % save_every == save_every -1:
        if log_name is None:
            log_name = time.strftime("%Y%m%d_%H%M%S")
        torch.save(model.model.state_dict(), './log/hc_{}_{}_model.pth'.format(log_name, ep))
        torch.save(acc_R, './log/hc_{}_reward.pth'.format(log_name))
        print('model saved at'+log_name)
    model.reset()
    model.train()