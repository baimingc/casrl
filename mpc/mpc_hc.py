import numpy as np
from .optimizers import RandomOptimizer, CEMOptimizer
import copy
import math

class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, mpc_config):
        self.constraint = mpc_config["constraint"]
        self.prior_safety = mpc_config["prior_safety"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = np.array(conf["action_low"]) 
        self.action_high = np.array(conf["action_high"]) 
        self.action_dim = conf["action_dim"]
        self.popsize = conf["popsize"]
        self.action_cost = conf["action_cost"]
        self.x_dot_cost = conf["x_dot_cost"]
        self.particle = conf["particle"]

        self.init_mean = np.array([conf["init_mean"]] * self.horizon)
        self.init_var = np.array([conf["init_var"]] * self.horizon)

        if len(self.action_low) == 1: # auto fill in other dims
            self.action_low = np.tile(self.action_low, [self.action_dim])
            self.action_high = np.tile(self.action_high, [self.action_dim])
        
        self.optimizer = MPC.optimizers[self.type](sol_dim=self.horizon*self.action_dim,
                                                   popsize=self.popsize,
                                                   upper_bound=np.array(conf["action_high"]),
                                                   lower_bound=np.array(conf["action_low"]),
                                                   max_iters=conf["max_iters"],
                                                   num_elites=conf["num_elites"],
                                                   epsilon=conf["epsilon"],
                                                   alpha=conf["alpha"])

        self.optimizer.setup(self.ar_cost_function)
        self.reset()
        
        self.constraint_reward = -100
        

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        #print('set init mean to 0')
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])

    def act(self, model, state, ground_truth=False):
        '''
        :param state: task, model, (numpy array) current state
        :return: (float) optimal action
        '''
        self.model = model
        self.state = state
        self.ground_truth = ground_truth

        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        if self.type == "CEM":
            self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        action = soln[:self.action_dim]
        
        return action

    def preprocess(self, state):
        return state
    
    
    def ar_cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """

        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))

        costs = np.zeros(self.popsize*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0)
        state_prior = state
            
        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)

            state_predict = self.model.predict(state, action)
            state_next = state_predict + state

            cost = self.ar_cost(state_next, action)  # compute cost
            state = state_next

            costs += cost * self.gamma**t
            
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def ar_cost(self, state, action, env_cost=False, obs=True):
        """
        Calculate the assistrobot env cost given the state

        Parameters:
        ----------
            @param numpy array - state : size should be (batch_size x state dim)
            @param numpy array - action : size should be (batch_size x action dim)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """

        distance_weight = 1.0
        action_weight = 0.01
        food_reward_weight = 1.0
        if self.constraint:
            contact_weight = 100
        else:
            contact_weight = 0

        distance_mouth_target = np.linalg.norm(state[:, 6:9], axis=1)
        reward_action = -np.sum(np.square(action), axis=1) # Penalize actions
        
        spilled = (abs(state[:,3] - 1.46) > 0.15) | (abs(state[:,4]) > 0.1)

        reached = distance_mouth_target < 0.03
        reward_food = reached * 20 + spilled * (-100)
        
        contact = state[:,-1] > 0.5
        
        
        reward = -distance_weight*distance_mouth_target + action_weight*reward_action + food_reward_weight*reward_food - contact_weight*contact
        cost = -reward
        return cost

    def ar_cost_prior(self, state, action, env_cost=False, obs=True):
        """
        Calculate the constraint violation of prior model

        Parameters:
        ----------
            @param numpy array - state : size should be (batch_size x state dim)
            @param numpy array - action : size should be (batch_size x action dim)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        
        violation = self.constraint_violated(state)
        reward = violation * self.constraint_reward
        cost = -reward

        return cost

