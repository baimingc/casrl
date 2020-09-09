'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-04-06 20:40:15
@LastEditTime: 2020-04-09 19:10:40
@Description: 
'''

import sys
import os
import copy

import numpy as np
import tensorflow as tf

# add the path of learning_to_adapt
sys.path.append(os.getcwd()+'/baselines/learning_to_adapt')
from learning_to_adapt.dynamics.meta_mlp_dynamics import MetaMLPDynamicsModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Config_Parser(object):
    def __init__(self, grbal_config):
        self.valid_split_ratio = grbal_config['valid_split_ratio']
        self.rolling_average_persitency = grbal_config['rolling_average_persitency']
        self.obs_space_dims = grbal_config['obs_space_dims']
        self.action_space_dims = grbal_config['action_space_dims']
        self.max_path_length = grbal_config['max_path_length']
        self.n_itr = grbal_config['n_itr']

        self.meta_batch_size = grbal_config['meta_batch_size']
        self.adapt_batch_size = grbal_config['adapt_batch_size']
        self.hidden_nonlinearity_model = grbal_config['hidden_nonlinearity_model']
        self.learning_rate = grbal_config['learning_rate']
        self.inner_learning_rate = grbal_config['inner_learning_rate']
        self.hidden_sizes_model = tuple(grbal_config['hidden_sizes_model'])
        self.dynamics_model_max_epochs = grbal_config['dynamics_model_max_epochs']


class GrBAL(object):
    name = 'GrBAL'

    def __init__(self, grbal_config):
        config = Config_Parser(grbal_config=grbal_config)

        self.dynamics_model = MetaMLPDynamicsModel(
            name=self.name,
            obs_space_dims=config.obs_space_dims,
            action_space_dims=config.action_space_dims,
            meta_batch_size=config.meta_batch_size,
            inner_learning_rate=config.inner_learning_rate,
            learning_rate=config.learning_rate,
            hidden_sizes=config.hidden_sizes_model,
            valid_split_ratio=config.valid_split_ratio,
            rolling_average_persitency=config.rolling_average_persitency,
            hidden_nonlinearity=config.hidden_nonlinearity_model,
            batch_size=config.adapt_batch_size
        )

        self.n_itr = config.n_itr
        self.dynamics_model_max_epochs = config.dynamics_model_max_epochs
        self.max_path_length = config.max_path_length # each rollout should have the max path length
        self.adapt_batch_size = config.adapt_batch_size

        # initialize adapt data buffer
        self.adapt_obs = []
        self.adapt_act = []
        self.adapt_next_obs = []

        # initialize training data buffer
        self.train_obs = {}
        self.train_act = {}
        self.train_next_obs = {}

        self.sess = tf.Session()

    def data_process(self, data):
        path_index = data[0]
        s = data[1]
        a = data[2]
        s_n = data[3]

        # data shape should be [episode_numper, episode_length, observation size]
        if path_index not in self.train_obs.keys():
            self.train_obs[path_index] = [s]
            self.train_act[path_index] = [a]
            self.train_next_obs[path_index] = [s_n]
        else:
            self.train_obs[path_index].append(s)
            self.train_act[path_index].append(a)
            self.train_next_obs[path_index].append(s_n)

    def fit(self):
        # prepare the data
        obs = None
        act = None
        obs_next = None
        for k_i in self.train_obs.keys():
            if obs is None:
                obs = np.array(self.train_obs[k_i])[None]
                act = np.array(self.train_act[k_i])[None]
                obs_next = np.array(self.train_next_obs[k_i])[None]
            else:
                obs = np.concatenate((obs, np.array(self.train_obs[k_i])[None]), axis=0)
                act = np.concatenate((act, np.array(self.train_act[k_i])[None]), axis=0)
                obs_next = np.concatenate((obs_next, np.array(self.train_next_obs[k_i])[None]), axis=0)

        self.dynamics_model.fit(obs, act, obs_next, epochs=self.dynamics_model_max_epochs)

    def reset_train_data(self):
        self.train_obs = {}
        self.train_act = {}
        self.train_next_obs = {}

    def reset_adapt_data(self):
        self.adapt_obs = []
        self.adapt_act = []
        self.adapt_next_obs = []

    def adapt(self, data):
        if len(self.adapt_obs) > self.adapt_batch_size:
            p_i, state, action, state_next = data
            self.adapt_obs.append(state)
            self.adapt_act.append(action)
            self.adapt_next_obs.append(state_next)

            self.dynamics_model.switch_to_pre_adapt()
            self.dynamics_model.adapt([np.array(self.adapt_obs)], [np.array(self.adapt_act)], [np.array(self.adapt_next_obs)])

    def predict(self, s, a):
        # in GrBAL model, the predict is delta_state, but it has added the previous state
        s_next = self.dynamics_model.predict(s, a)
        # in our MPC module, the prediction should be s_next-s
        return s_next - s

    def meta_train(self, meta_task, mpc_controller):
        for t_i in range(self.n_itr):
            # in one iteration, we should collect rollout data
            for p_i, subtask in enumerate(meta_task):
                current_length = 0
                finish_one_path = False
                while not finish_one_path:
                    state = subtask.reset()
                    self.reset_adapt_data()
                    action = subtask.action_space.sample()
                    state_next, reward, done, info = subtask.step(action)
                    while not done:
                        if t_i == 0:
                            # use random policy before the model is fit
                            action = subtask.action_space.sample()
                        else:
                            # create a new adaptive model
                            # Comment this out and it won't adapt during rollout
                            self.adapt([p_i, state, action, state_next])
                            # use model prediction and mpc policy
                            action = np.array([mpc_controller.act(task=subtask, model=self, state=state)])
                        
                        # add the data to databuffer
                        self.data_process([p_i, state, action, state_next])
                        # interact with env
                        state_next, reward, done, info = subtask.step(action)
                        state = copy.deepcopy(state_next)
                        current_length += 1
                        
                        # stop one path
                        if current_length >= self.max_path_length:
                            finish_one_path = True
                            break
                print('Finish one episode...')
                
            # train the meta model 
            self.fit()
            self.reset_train_data()
            print('Iteration: {}'.format(t_i))
