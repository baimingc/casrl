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

        self.optimizer.setup(self.mg_cost_function)
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
    
    
    def mg_cost_function(self, actions):
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

            cost = self.mg_cost(state_next, action)  # compute cost
            state = state_next

            costs += cost * self.gamma**t
            
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def mg_cost(self, state, action, env_cost=False, obs=True):
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
        COLLISION_REWARD: float = -1
        RIGHT_LANE_REWARD: float = 0.1
        HIGH_SPEED_REWARD: float = 0.02
        MERGING_SPEED_REWARD: float = -0.5
        LANE_CHANGE_REWARD: float = -0.05

        crashed = abs(state[:,4]) < 0.05
        right_lane = abs(state[:,1]) < 0.1
        speed = state[:,2]


        reward = COLLISION_REWARD * crashed \
                 + RIGHT_LANE_REWARD * right_lane \
                 + HIGH_SPEED_REWARD * speed
        cost = -reward
        return cost

    def mg_cost_prior(self, state, action, env_cost=False, obs=True):
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

