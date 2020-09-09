'''
@Author:
@Email: 
@Date: 2020-04-01 01:26:48
@LastEditTime: 2020-04-16 00:40:07
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

import gym
sys.path.append('./envs/cartpole-envs')
sys.path.append('./envs/highway-env')
import cartpole_envs
#import highway_env

from utils import plot_reward, plot_index
from mpc.mpc_cp import MPC

# all models
from dpgpmm.DPGPMM import DPGPMM
# from baselines.SingleGP import SingleGP
# from baselines.SingleSparseGP import SingleSparseGP
from baselines.NN import NN


def prepare_dynamics(gym_config):
    dynamics_name = gym_config['dynamics_name']
    seed = gym_config['seed']
    dynamics_set = []
    for i in range(len(dynamics_name)):
        env = gym.make(dynamics_name[i])
        # env.seed(seed)
        dynamics_set.append(gym.make(dynamics_name[i]))
    
    # use pre-defined env sequence
    task = [dynamics_set[i] for i in gym_config['task_dynamics_list']]
    return task


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)


if __name__ == '__main__':
    # dynamic model configuration
    # config = load_config('config_DPGP_MBRL.yml')
    config = load_config('./config/config_stable.yml')
    dpgp_config = config['DPGP_config']
    gp_config = config['SingleGP_config']
    sparse_gp_config = config['SingleSparseGP_config']
    nn_config = config['NN_config']
    mpc_config = config['mpc_config']
    gym_config = config['gym_config']
    render = gym_config['render']

    # initialize the mixture model
    model = DPGPMM(dpgp_config=dpgp_config)
    # model = SingleSparseGP(sparse_gp_config=sparse_gp_config)
    # model = SingleGP(gp_config=gp_config)
    # model = NN(NN_config=nn_config)
    logger.info('Using model: {}', model.name)

    # initial MPC controller
    mpc_controller = MPC(mpc_config=mpc_config)

    # prepare task
    # the task is solved, if each dynamic is solved
    task = prepare_dynamics(gym_config)

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

    while (not task_solved) and (task_epi < gym_config['task_episode']):
        task_epi += 1
        time_task_0 = time.time()
        if total_count == 0:
            # for the first step, add one data pair with random policy as initialization
            state = task[0].reset()
            action = task[0].action_space.sample()
            state_next, reward, done, info = task[0].step(action)
            model.fit(data=[0, state, action, state_next-state])
            label_list.append(0)

        # for other steps, run DPGP MBRL
        # Different sub-tasks share the same action space
        # Note that the subtask_index is unknown to the model, it's for debugging
        task_r = 0
        for subtask_index in range(len(task)):
            for epi in range(gym_config['subtask_episode']): # each subtask contains a fixed number of episode
                print('subtask: ', subtask_index, ', epi: ', epi)
                time_subtask_0 = time.time()
                acc_reward = 0
                state = task[subtask_index].reset()
                for step in range(gym_config['subtask_episode_length']):
                    if render:
                        task[subtask_index].render()
                    total_count += 1
                    label_list.append(subtask_index)

                    # MPC policy
                    start_1 = time.time()
                    action = np.array([mpc_controller.act(task=task[subtask_index], model=model, state=state)])
                    start_2 = time.time()

                    # Random Policy
                    # action = task[subtask_index].action_space.sample()

                    # interact with env
                    state_next, reward, done, info = task[subtask_index].step(action)
                    acc_reward += reward
                    # logger.info('acc_reward : {}', acc_reward)
                    start_3 = time.time()

                    # train the model
                    # todo: if not train the model, need to fix the model index used
                    # if task_epi <= 5:
                    #     model.fit(data=[subtask_index, state, action, state_next-state])

                    # when reach some kind of metric, stop training, only inference
                    if model.name == 'DPGPMM':
                        model.fit(data=[subtask_index, state, action, state_next - state], comp_trainable=comp_trainable)
                        # print('index ', model.index)
                        if len(subtask_succ_count) < len(model.DP_mix.comps):
                            print('add a new comp')
                            subtask_succ_count.append(0)
                            comp_trainable.append(1)
                        elif len(subtask_succ_count) > len(model.DP_mix.comps):
                            print('del the last component')
                            subtask_succ_count.pop()
                            comp_trainable.pop()
                    else:
                        if trainable:
                            model.fit(data=[subtask_index, state, action, state_next - state])

                    state = copy.deepcopy(state_next)
                    start_4 = time.time()

                    # print('mpc: {}, env: {}, model: {}'.format(start_2-start_1, start_3-start_2, start_4-start_3))

                    if done:
                        print('-------------------------------------------------')
                        print('Episode finished: Fail, time: ', time.time()-time_subtask_0, ' with reward: ', acc_reward)
                        print('-------------------------------------------------')
                        subtask_list.append(subtask_index)
                        subtask_reward.append(acc_reward)
                        task_r += acc_reward
                        if not model.name == 'DPGPMM':
                            if len(subtask_succ_count) < subtask_index + 1:
                                subtask_succ_count.append(0)
                        break
                    elif acc_reward >= 0.975*gym_config['subtask_episode_length']:
                        subtask_solved[subtask_index] = True
                        print('-------------------------------------------------')
                        print('Episode finished: Success!!!!, time: ', time.time()-time_subtask_0)
                        print('-------------------------------------------------')
                        subtask_list.append(subtask_index)
                        subtask_reward.append(acc_reward)
                        task_r += acc_reward

                        # record succ rate
                        # todo: check whether assigning index like this is reasonable
                        if model.name == 'DPGPMM':
                            subtask_succ_count[model.DP_mix.assigns[len(model.DP_mix.data) - 1]] += 1
                        else:
                            if len(subtask_succ_count) < subtask_index + 1:
                                subtask_succ_count.append(1)
                            else:
                                subtask_succ_count[subtask_index] += 1
                        break
                    
                if model.name == 'DPGPMM':
                    print('subtask_succ_count: ', subtask_succ_count)
                    # todo: check the training termination criterion right or not
                    for i in range(len(subtask_succ_count)):
                        if subtask_succ_count[i] >= 10:
                            comp_trainable[i] = 0
                else:
                    print('subtask_succ_count: ', subtask_succ_count)
                    # todo: check the training termination criterion right or not
                    all_solve = 0
                    for i in range(len(subtask_succ_count)):
                        if subtask_succ_count[i] >= 10:
                            all_solve += 1
                    if all_solve == 4:
                        trainable = False
                if render:
                    task[subtask_index].close()
        task_reward.append(task_r)
        time_task = time.time() - time_task_0
        if np.sum(subtask_solved*1) == len(subtask_solved):
            # task_solved = True
            logger.info('Solve all subtasks!')
            # break

        if task_epi % 1 == 0:
            # record the reward
            plot_reward(subtask_list, subtask_reward,
                        name=model.name + '_' + 'subtask_reward_CartPole_' + str(task_epi), xlabel='subtask', y=195)
            plot_reward(range(len(task_reward)), task_reward,
                        name=model.name + '_' + 'task_reward_CartPole_' + str(task_epi), xlabel='episode', y=1560, scatter=False)

            print('***************************')
            print('task_episode: ', task_epi, ' time: ', time_task)
            if model.name == 'DPGPMM':
                numbers = []
                for comp in model.DP_mix.comps:
                    numbers.append(comp.n)
                print('data in each component: ', numbers)
                print('***************************')
                with open('./misc/data.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(numbers) + ' ' + str(task_r) + '\n')
                    # f.write('task_episode: ' + str(task_epi) + ' time: ' + str(time_task)
                    #         + ' data in each component: ' + str(numbers))
                # record the updated assignments
                predict_list = []
                for i in range(len(model.DP_mix.data)):
                    predict_list.append(model.DP_mix.assigns[i])
                plot_index(predict_list, label_list, name='Cluster Result with CartPole ' + str(task_epi))
            else:
                with open('./misc/data_' + model.name + '.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(task_r) + '\n')
                print('***************************')
