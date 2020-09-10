import time, datetime
import os, sys
import numpy as np
import yaml
import gym
import torch

from mpc.mpc_mg import MPC
from model.NP import NP

sys.path.append('./envs/highway-env')
import highway_env


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)
        
config = load_config('config/config_merge.yml')
mpc_config = config['mpc_config']
prior_safety = mpc_config['prior_safety']
np_config = config['NP_config']

model = NP(NP_config=np_config)
mpc_controller = MPC(mpc_config=mpc_config)
env = gym.make("merge-v0")


"""testing the model with MPC while training """
test_episode = 200
save_every = 5
log = []
log_name = None
model.reset()

for ep in range(test_episode):
    task_steps = 0

    obs = env.reset()
    O, A, R, acc_reward, done = [], [], [], 0, False
    mpc_controller.reset()
    i = 0
    while not done:
        i+= 1
        if task_steps > 0:
            action = np.array(mpc_controller.act(model=model, state=obs, ground_truth=True))
        else:
            action = np.array([0.0, 0.0])
        obs_next, reward, done, violation = env.step(action)
        task_steps += 1
        A.append(action)
        O.append(obs_next)
        R.append(reward)
        model.data_process([0, obs, action, obs_next - obs])
        obs = obs_next
        acc_reward += reward
    print('step: ', i, 'acc_reward: ', acc_reward)
    env.close()

    if done:
        samples = {
            "obs": np.array(O),
            "actions": np.array(A),
            "rewards": np.array(R),
            "reward_sum": acc_reward,
        }
        log.append(samples)
        if ep % save_every == save_every -1:
            if log_name is None:
                log_name = time.strftime("%Y%m%d_%H%M%S")
            torch.save(model.model.state_dict(), './log/mg_{}_{}_model.pth'.format(log_name, ep))
            torch.save(log, './log/mg_{}_log.pth'.format(log_name))
            print('model saved at'+log_name)
    model.reset()
    model.train()