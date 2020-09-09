'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 10:42:50
@LastEditTime: 2020-04-13 19:32:02
@Description:
'''

import numpy as np
from utils import *
from models.nn import DynamicModel
from optimizers import RandomOptimizer, CEMOptimizer
from tqdm import trange, tqdm


class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, env, config, model):
        self.env = env
        mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = conf["action_low"]
        self.action_high = conf["action_high"]
        self.popsize = conf["popsize"]
        self.model = model
        
        self.init_mean = np.array([conf["init_mean"]]*self.horizon)
        self.init_var = np.array([conf["init_var"]]*self.horizon)

        self.optimizer = MPC.optimizers[self.type](sol_dim=self.horizon,
                                                   popsize=self.popsize,
                                                   upper_bound=self.action_high,
                                                   lower_bound=self.action_low,
                                                   max_iters=conf["max_iters"],
                                                   num_elites=conf["num_elites"],
                                                   epsilon=conf["epsilon"],
                                                   alpha=conf["alpha"])

        self.optimizer.setup(self.cost_function)

    def act(self, state):
        '''
        :param state: (numpy array) current state
        :return: (float) optimal action
        '''
        self.state = state
        solution, var = self.optimizer.obtain_solution(self.init_mean, self.init_var)[0]
        return solution

    def cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions

        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        def angle_normalize(x):
            return (((x+np.pi) % (2*np.pi)) - np.pi)

        batch_size = actions.shape[0]
        costs = np.zeros(batch_size)
        state = np.repeat(self.state.reshape(1,-1), batch_size, axis=0)
        for t in range(self.horizon):
            action = actions[:,t].reshape(-1,1) # numpy array (batch_size x action dim)
            state_next = self.model.predict(state, action)+state  # numpy array (batch_size x state dim)
            state_next[:,0] = angle_normalize(state_next[:,0])
            cost = self.pendulum_cost(state_next, action) # compute cost
            costs = costs + cost*self.gamma
            state = state_next
        return costs

    def pendulum_cost(self, state, action):
        """
        Calculate the pendulum env cost given the state

        Parameters:
        ----------
            @param numpy array - state : size should be (batch_size x state dim)
            @param numpy array - action : size should be (batch_size x action dim)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        
        th = state[:,0]
        thdot = state[:,1]
        u = action[:,0]
        cost = th**2 + .1*thdot**2 + .001*(u**2)
        #reward = -np.sqrt(2*(1.0 - cos_theta))
        #cost = - reward
        return cost

    def mpc_itr(self, episodes=100, max_step = 200, render=False):
        """
        Collect training dynamic data from mpc controller

        Parameters:
        ----------
            @param int - episodes : determine how many episodes data to collect
            @param int - max_step : max steps for each episode
            @param bool - render : render the env

        Return:
        ----------
            @param list of numpy array - state_action_pairs : list of training data (state + action)
            @param list of numpy array - delta_states : list of label (next_state - state)
        """
        state_action_pairs, delta_states = [], []
        training_set = []
        
        t = trange(episodes)
        rewards = []
        for epi in t:
            state = self.env.reset()
            if render:
                self.env.render()
            episode_reward = []
            for step in range(max_step):
                action = np.array([self.act(state)])
                state_next, reward, done, _ = self.env.step(action)
                state_action_pair = np.concatenate((state, action))
                delta_state = state_next-state
                state_action_pairs.append(state_action_pair)
                delta_states.append(delta_state)
                state = state_next
                episode_reward.append(reward)
                rewards.append(reward)
                if render:
                    self.env.render()
            t.set_description(f"reward: {np.mean(episode_reward):.2f}")
            t.refresh()
        tqdm.write("[INFO] mean reward during this mpc iteration: %.2f"%(np.mean(rewards)))
        return state_action_pairs, delta_states