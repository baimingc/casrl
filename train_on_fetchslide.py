'''
@Author:
@Email: 
@Date: 2020-04-01 01:26:48
@LastEditTime: 2020-05-05 23:25:02
@Description: 
'''

import time, datetime
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
sys.path.append('./envs/fetchslide-env')
import fetchslide_envs

from utils import plot_reward, plot_index, dumb_reward_plot, save_test_succ_rate, Noise
# use fetch slide mpccontroller
from mpc.mpc_fs import MPC

# all models
from dpgpmm.DPGPMM import DPGPMM
from baselines.SingleGP import SingleGP
from baselines.SingleSparseGP import SingleSparseGP
from baselines.NN import NN
from baselines.SNN import SNN
from dpgpmm.SDPGPMM import SDPGPMM


def prepare_dynamics(gym_config):
    dynamics_name = gym_config['dynamics_name']
    #seed = gym_config['seed']
    dynamics_set = []
    for i in range(len(dynamics_name)):
        # env = gym.make(dynamics_name[i])
        # env.seed(seed)
        dynamics_set.append(gym.make(dynamics_name[i]))
    
    # use pre-defined env sequence
    task = [dynamics_set[i] for i in gym_config['task_dynamics_list']]
    return task


def mpc_reward_visualization(state, action):
    desired_goal = state['desired_goal']
    observation = state['observation']

    grip_pos = observation[0]
    object_pos = observation[3]

    vec_1 = object_pos - grip_pos
    vec_2 = object_pos - desired_goal[0]

    # when the object goes too far away, punish it
    overshot_penality = 1.0 * (vec_2 > 0)

    cost_near = np.abs(vec_1)
    cost_dist = np.abs(vec_2)
    cost = 1.0 * cost_dist + 0.5 * cost_near + overshot_penality

    cost_ctrl = action[0]**2
    cost += 0.01 * cost_ctrl

    return cost


def fetchslide_state_process(action, state, state_next):
    desired_goal = state['desired_goal'][0:2]
    _action = action[1:2]
    
    observation = state['observation']
    grip_pos = observation[0:2]
    object_pos = observation[3:5]
    object_velp = observation[14:17]
    grip_velp = observation[20:23]

    _state = np.concatenate((grip_pos[1:2], grip_velp[1:2], object_pos[1:2], object_velp[1:2]), axis=0)

    observation = state_next['observation']
    grip_pos = observation[0:2]
    object_pos = observation[3:5]
    object_velp = observation[14:17]
    grip_velp = observation[20:23]

    _state_next = np.concatenate((grip_pos[1:2], grip_velp[1:2], object_pos[1:2], object_velp[1:2]), axis=0)
    return _action, _state, _state_next


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)


if __name__ == '__main__':
    # dynamic model configuration
    # config = load_config('config_DPGP_MBRL.yml')
    config = load_config('./config/config_fetchslide.yml')
    dpgp_config = config['DPGP_config']
    gp_config = config['SingleGP_config']
    sparse_gp_config = config['SingleSparseGP_config']
    nn_config = config['NN_config']
    mpc_config = config['mpc_config']
    gym_config = config['gym_config']
    render = gym_config['render']

    # initialize the mixture model
    #model = SDPGPMM(dpgp_config=dpgp_config)
    #model = DPGPMM(dpgp_config=dpgp_config)
    #model = SingleSparseGP(sparse_gp_config=sparse_gp_config)
    model = SingleGP(gp_config=gp_config)
    #model = SNN(NN_config=nn_config)
    #model = NN(NN_config=nn_config)
    logger.info('Using model: {}', model.name)

    # initial MPC controller
    mpc_controller = MPC(mpc_config=mpc_config)
    action_noise = Noise(mu=np.zeros(mpc_config['CEM']['action_dim']), sigma=0.05)

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
    log = []
    succ_rate_list = []
    log_name = None
    test_iter = 1000
    test_length = 40
    NN_train_interval = 1

    while (not task_solved) and (task_epi < gym_config['task_episode']):
        task_epi += 1
        time_task_0 = time.time()
        if total_count == 0:
            # for the first step, add one data pair with random policy as initialization
            state = task[0].reset()
            action = task[0].action_space.sample()
            state_next, reward, done, info = task[0].step(action)

            # process data (parse state and delete one dimension of action)
            _action, _state, _state_next = fetchslide_state_process(action, state, state_next)
            # train the model
            if model.name in ['NN', 'SNN']:
                model.data_process(data=[0, _state, _action, _state_next - _state])
            elif model.name in ['DPGPMM', 'SDPGPMM', 'SingleGP', 'SingleSparseGP']:
                model.fit(data=[0, _state, _action, _state_next - _state])
            else:
                RuntimeError('No such a model')
            label_list.append(0)

        # for other steps, run DPGP MBRL
        # Different sub-tasks share the same action space
        # Note that the subtask_index is unknown to the model, it's for debugging
        task_r = 0
        for subtask_index in range(len(task)):
            for epi in range(gym_config['subtask_episode']): # each subtask contains a fixed number of episode

                # test stage
                if (epi+1) % test_iter == 0:
                    succ_rate = 0
                    for t_i in range(test_length):
                        state = task[subtask_index].reset()
                        mpc_controller.reset()
                        done = False
                        print('Testing: ', t_i)
                        while not done:
                            if render: task[subtask_index].render()
                            action = mpc_controller.act(task=task[subtask_index], model=model, state=state)
                            state_next, reward, done, info = task[subtask_index].step(action)
                            if info['is_success']:
                                succ_rate += 1
                    succ_rate /= test_length
                    succ_rate_list.append(succ_rate)
                    print('-------------------------------------------------')
                    print('Test success rate: ', succ_rate)
                    print('-------------------------------------------------')
                    save_test_succ_rate(succ_rate_list)
                            
                # train stage
                O, A, R, acc_reward, done = [], [], [], -np.inf, False
                print('subtask: ', subtask_index, ', epi: ', epi)
                time_subtask_0 = time.time()

                state = task[subtask_index].reset()
                O.append(state)
                # reset the controller at the beginning of each new dynamic
                mpc_controller.reset()

                while not done:
                    if render: task[subtask_index].render()

                    total_count += 1
                    label_list.append(subtask_index)

                    # MPC policy
                    action = mpc_controller.act(task=task[subtask_index], model=model, state=state)# + action_noise.get_noise()

                    # mpc reward visualization
                    #mpc_reward = mpc_reward_visualization(state, action)

                    # interact with env
                    state_next, reward, done, info = task[subtask_index].step(action)
                    #acc_reward = reward if reward > acc_reward else acc_reward

                    A.append(action)
                    O.append(np.concatenate((state_next['observation'], state_next['desired_goal'])))
                    R.append(reward)

                    # process data (parse state and delete one dimension of action)
                    _action, _state, _state_next = fetchslide_state_process(action, state, state_next)

                    ###################
                    _state_pre = model.predict(_state[None], _action[None])
                    print('action ', action)
                    print('input: ',  _state)
                    print('reward: %.4f' % reward)
                    print('predict    ', _state_pre[0])
                    print('observation', _state_next-_state)
                    #print('loss ', np.sum((_state_pre[0]-(_state_next-_state))**2))
                    ###################

                    # train the model
                    if model.name in ['NN', 'SNN'] and not done:
                        # Except DPGP, train the model at the end of one episode (even not every episode)
                        model.data_process(data=[subtask_index, _state, _action, _state_next-_state])
                    else:
                        model.fit(data=[subtask_index, _state, _action, _state_next-_state])

                    # update state
                    state = copy.deepcopy(state_next)

                    if done:
                        acc_reward = reward
                        samples = {
                            "obs": np.array(O),
                            "actions": np.array(A),
                            "rewards": np.array(R),
                            "reward_sum": acc_reward,
                        }
                        log.append(samples)
                        if log_name is None:
                            log_name = datetime.datetime.now()
                        path = './misc/log/' + log_name.strftime("%d-%H-%M") + '.npy'
                        np.save(path, log, allow_pickle=True)
                        dumb_reward_plot(path, PREFIX='Fetchslide', xlim=[0, 100], ylim=[-0.5, 0], y_line1=-0.05, y_line2=-0.05)

                        print('-------------------------------------------------')
                        print('Episode finished, time:', time.time()-time_subtask_0, 'with minimal reward:', acc_reward)
                        print('-------------------------------------------------')
                        subtask_list.append(subtask_index)
                        subtask_reward.append(acc_reward)

                        if info['is_success']:
                            subtask_solved[subtask_index] = True
                            print('-------------------------------------------------')
                            print('Episode finished: Success!!!!, time: ', time.time()-time_subtask_0)
                            print('-------------------------------------------------')
                            subtask_list.append(subtask_index)
                            subtask_reward.append(acc_reward)
                            task_r += acc_reward

        task_reward.append(task_r)
        time_task = time.time() - time_task_0
        if np.sum(subtask_solved*1) == len(subtask_solved):
            logger.info('Solve all subtasks!')

        if task_epi % 1 == 0:
            # record the reward
            #plot_reward(subtask_list, subtask_reward, name=model.name + '_' + 'episode_reward' + str(task_epi), xlabel='subtask', y=195)
            #plot_reward(range(len(task_reward)), task_reward, name=model.name + '_' + 'final_reward_' + str(task_epi), xlabel='episode', y=1560, scatter=False)

            print('***************************')
            print('task_episode: ', task_epi, ' time: ', time_task)
            if model.name in ['DPGPMM', 'SDPGPMM']:
                numbers = []
                for comp in model.DP_mix.comps:
                    numbers.append(comp.n)
                print('data in each component: ', numbers)
                with open('./misc/data_FetchSlide.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(numbers) + ' ' + str(task_r) + '\n')
                # record the updated assignments
                predict_list = []
                for i in range(len(model.DP_mix.data)):
                    predict_list.append(model.DP_mix.assigns[i])
                plot_index(predict_list, label_list, name='Cluster_Result_with_FetchSlide_' + str(task_epi) + '.png')
            else:
                with open('./misc/data_FetchSlide_' + model.name + '.txt', 'a') as f:
                    f.write(str(task_epi) + ' ' + str(time_task) + ' ' + str(task_r) + '\n')
            print('***************************')
