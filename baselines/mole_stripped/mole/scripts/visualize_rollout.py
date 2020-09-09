
from tf.envs.base import TfEnv

import numpy as np
from rllab.misc import tensor_utils
import time
import tensorflow as tf
import copy
import joblib
import matplotlib.pyplot as plt
import argparse
import yaml
import os.path as osp

from rllab.envs.normalized_env import normalize

from mole.utils import create_env


def main(args):

    ## Create env
    saved_config = yaml.load(open(osp.join(args.filename, 'config.yaml')))
    env, _ = create_env(saved_config['env']['name'], args.task)

    ## Load info
    prefix = args.task + '_n' + str(args.thetaResetChoice) + '_rollout' + str(args.whichRollout)

    starting_state = np.load(args.filename + prefix + '_startingstate.npy', allow_pickle=True)
    actions = np.load(args.filename + prefix + '_actions.npy', allow_pickle=True)
    probabilities = np.load(args.filename + prefix + '_probabilities.npy', allow_pickle=True)
    ubs = np.load(args.filename + prefix + '_ubs.npy', allow_pickle=True)
    saved_rewards = np.load(args.filename + prefix + '_rewards.npy', allow_pickle=True)

    print("\n\nNUM STEPS: ")
    print(actions.shape[0])
    print(ubs)
    # each entry of probabilities
    # corresponds to a timestep, 
    # where the entry is an array like [P(theta0), P(theta1), ...]
    print(len(probabilities)) 

    # live plotting of probability distribution
    do_live_plot = False
    if args.thetaResetChoice == 5:
        if args.render:
            do_live_plot = True
    
    ## Setup before rollout
    dt = env.wrapped_env.wrapped_env.model.opt.timestep
    dt_updated_for_viz = dt / args.speedup
    # env._wrapped_env._wrapped_env.reset_task()
    env.reset(init_state = starting_state)
    if args.render: env.render()

    ## Execute saved actions to see rollout
    reward = []
    for i in range(actions.shape[0]):

        if(i%100==0): print(i)

        # take action + record results
        env.wrapped_env.wrapped_env.step_num=i
        next_o, r, d, env_info = env.step(actions[i])
        if i>=ubs: reward.append(r)

        # live plot of the task probabilities
        if do_live_plot:
            if(i%10==0):
                if i>=ubs:
                    plt.clf()
                    plt.ylim(0,1)
                    plt.bar(np.arange(len(probabilities[i-ubs])), probabilities[i-ubs])
                    plt.pause(dt_updated_for_viz/100)

        # render
        if args.render:
            if i>0:
                env.render()
                time.sleep(dt_updated_for_viz)

    if args.render: env.render(close=True)

    print("\n\nROLLOUT RECORDED REWARD: ", np.sum(saved_rewards[ubs:]))
    print("ROLLOUT ACHIEVED REWARD: ", np.sum(reward),"\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-n', '--thetaResetChoice', type=int, default=5)
    parser.add_argument('-r', '--whichRollout', type=int, default=0)
    parser.add_argument('-speedup', '--speedup', type=int, default=1)
    parser.add_argument('-render', '--render', action='store_true')
    args = parser.parse_args()
    main(args)