import time, datetime
import os, sys
import numpy as np
import yaml
import gym
import torch

from mpc.mpc_cp import MPC
from model.NP import NP

sys.path.append('./envs/cartpole-envs')
import cartpole_envs

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)
        
config = load_config('config/config_cartpole.yml')
mpc_config = config['mpc_config']
prior_safety = mpc_config['prior_safety']
np_config = config['NP_config']

model = NP(NP_config=np_config)
mpc_controller = MPC(mpc_config=mpc_config)
env = gym.make("CartPoleSwingUpEnv-v0")


"""testing the model with MPC while training """
test_episode = 200
save_every = 5
log = []
log_name = None
model.reset()

for ep in range(test_episode):
    task_steps = 0
    m_p = np.random.uniform(0.2, 0.8)
    l = np.random.uniform(0.2, 0.8)
    env.unwrapped.m_p = m_p
    env.unwrapped.l = l

    obs = env.reset()
    O, A, R, acc_reward, done, V = [], [], [], 0, False, []
    mpc_controller.reset()
    i = 0
    while not done:
        i+= 1
        if task_steps > 0:
            action = np.array([mpc_controller.act(task=env, model=model, state=obs, ground_truth=True)])
        else:
            action = np.array([0.0])
        obs_next, reward, done, violation = env.step(action)
        task_steps += 1
        A.append(action)
        O.append(obs_next)
        R.append(reward)
        V.append(violation)
        model.data_process([0, obs, action, obs_next - obs])
        obs = obs_next
        acc_reward += reward
    print('pole_mass: ', m_p, 'pole_length: ', l, 'step: ', i, 'acc_reward: ', acc_reward, 'violation_rate: ', sum(V)/len(V))
    env.close()

    if done:
        samples = {
            "obs": np.array(O),
            "actions": np.array(A),
            "rewards": np.array(R),
            "reward_sum": acc_reward,
            "violation_rate": sum(V)/len(V)
        }
        log.append(samples)
        if ep % save_every == save_every -1:
            if log_name is None:
                log_name = time.strftime("%Y%m%d_%H%M%S")
            torch.save(model.model.state_dict(), './log/cp_{}_{}_model.pth'.format(log_name, ep))
            torch.save(log, './log/cp_{}_log.pth'.format(log_name))
            print('model saved at '+log_name)
    model.reset()
    model.train()