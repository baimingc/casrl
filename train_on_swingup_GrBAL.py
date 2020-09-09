'''
@Author:
@Email: 
@Date: 2020-04-01 01:26:48
@LastEditTime: 2020-04-16 00:22:48
@Description: 
'''

import time
import copy
import os
import sys
import warnings
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from loguru import logger
import yaml
from utils import plot_reward
import tensorflow as tf

import gym
sys.path.append('./envs/cartpole-envs')
import cartpole_envs

from mpc.mpc_cp import MPC

# all models
from baselines.learning_to_adapt.GrBAL import GrBAL


def prepare_dynamics(gym_config):
    dynamics_name = gym_config['dynamics_name']
    seed = gym_config['seed']
    dynamics_set = []
    for i in range(len(dynamics_name)):
        env = gym.make(dynamics_name[i])
        env.seed(seed)
        dynamics_set.append(gym.make(dynamics_name[i]))
    # use pre-defined env sequence
    adapt_task = [dynamics_set[i] for i in gym_config['task_dynamics_list']]
    rollout_list = [0, 1, 2, 3]
    meta_task = [dynamics_set[i] for i in rollout_list]
    return meta_task, adapt_task


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)


if __name__ == '__main__':
    # dynamic model configuration
    # config = load_config('config_DPGP_MBRL.yml')
    config = load_config('./config/config_swingup.yml')
    grbal_config = config['grbal_config']
    mpc_config = config['mpc_config']
    gym_config = config['gym_config']

    # initialize the mixture model
    model = GrBAL(grbal_config=grbal_config)
    logger.info('Using model: {}', model.name)

    # initial MPC controller
    mpc_controller = MPC(mpc_config=mpc_config)

    # prepare meta and adaptiive task
    meta_task, adapt_task = prepare_dynamics(gym_config)

    data_buffer = []
    subtask_list = []
    subtask_reward = []
    task_reward = []
    subtask_succ_count = [0]
    task_solved = False
    subtask_solved = [False, False, False, False]
    total_count = 0
    task_epi = 0

    with model.sess.as_default() as sess:
        """ Meta Learning Stage """
        sess.run(tf.initializers.global_variables())
        model.meta_train(meta_task, mpc_controller)

        """ Adapt Stage """
        #with tf.Session() as sess:
        while (not task_solved) and (task_epi < gym_config['task_episode']):
            task_epi += 1
            time_task_0 = time.time()
            if total_count == 0:
                # for the first step, add one data pair with random policy as initialization
                state = adapt_task[0].reset()
                action = adapt_task[0].action_space.sample()
                state_next, reward, done, info = adapt_task[0].step(action)

            task_r = 0
            for subtask_index in range(len(adapt_task)):
                for epi in range(gym_config['subtask_episode']): # each subtask contains a fixed number of episode
                    print('subtask: ', subtask_index, ', epi: ', epi)
                    time_subtask_0 = time.time()
                    acc_reward = 0
                    state = adapt_task[subtask_index].reset()
                    for step in range(gym_config['subtask_episode_length']):
                        total_count += 1

                        # MPC policy
                        model.adapt(data=[subtask_index, state, action, state_next])
                        action = np.array([mpc_controller.act(task=adapt_task[subtask_index], model=model, state=state)])

                        # interact with env
                        state_next, reward, done, info = adapt_task[subtask_index].step(action)
                        acc_reward += reward
                        state = copy.deepcopy(state_next)

                        if done:
                            print('-------------------------------------------------')
                            print('Episode finished, time: ', time.time()-time_subtask_0, ' with acc_reward: ', acc_reward, ' with final reward: ', reward)
                            print('-------------------------------------------------')
                            subtask_list.append(subtask_index)
                            subtask_reward.append(acc_reward)
                            task_r += acc_reward
                            if len(subtask_succ_count) < subtask_index + 1:
                                subtask_succ_count.append(0)
                            if reward >= 0.9:
                                subtask_solved[subtask_index] = True
                                print('-------------------------------------------------')
                                print('Episode finished: Success!!!!, time: ', time.time()-time_subtask_0)
                                print('-------------------------------------------------')
                                subtask_list.append(subtask_index)
                                subtask_reward.append(acc_reward)
                                task_r += acc_reward
                                # record succ rate
                                if model.name == 'DPGPMM':
                                    subtask_succ_count[model.DP_mix.assigns[len(model.DP_mix.data) - 1]] += 1
                                else:
                                    if len(subtask_succ_count) < subtask_index + 1:
                                        subtask_succ_count.append(1)
                                    else:
                                        subtask_succ_count[subtask_index] += 1
                    
                    print('subtask_succ_count: ', subtask_succ_count)
                    # todo: check the training termination criterion right or not
                    all_solve = 0
                    for i in range(len(subtask_succ_count)):
                        if subtask_succ_count[i] >= 10:
                            all_solve += 1
                    if all_solve == 4:
                        trainable = False

            task_reward.append(task_r)
            time_task = time.time() - time_task_0
            if np.sum(subtask_solved*1) == len(subtask_solved):
                # task_solved = True
                logger.info('Solve all subtasks!')
                # break

            if task_epi % 1 == 0:
                # record the reward
                plot_reward(subtask_list, subtask_reward, name=model.name+'_'+'subtask_reward_CartPole_'+str(task_epi))
                plot_reward(range(len(task_reward)), task_reward, name=model.name+'_'+'task_reward_CartPole_'+str(task_epi), xlabel='episode')
                with open('./misc/data_'+model.name+'.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + '\n')

                print('***************************')
                print('task_episode: ', task_epi, ' time: ', time_task)
                print('***************************')
