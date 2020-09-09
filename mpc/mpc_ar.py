import numpy as np
from tqdm import trange, tqdm
from .optimizers import RandomOptimizer, CEMOptimizer
import copy
import math

class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, mpc_config):
        # mpc_config = config["mpc_config"]
        self.constraint = mpc_config["constraint"]
        self.prior_safety = mpc_config["prior_safety"]
        self.only_prior_model = mpc_config["only_prior_model"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = np.array(conf["action_low"]) # array (dim,)
        self.action_high = np.array(conf["action_high"]) # array (dim,)
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

        # todo change the cost to the envirionment cost
        self.optimizer.setup(self.ar_cost_function)
        self.reset()
        
#         self.constraint = True #safe constraint
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
    
    def dumb_step(self, state, action):
        # Valid action
#         action = np.clip(action, -1.0, 1.0).flatten()
#         action *= self.force_mag
#         x = state[:, 0]
#         x_dot = state[:, 1]
#         c = state[:, 2]
#         s = state[:, 3]
#         theta_dot = state[:, 4]
#         theta = np.array([math.acos(c[i]) for i in range(len(c))])
#         theta[s < 0] = -theta[s < 0]
    
#         temp = 4 * action - 4 * self.b * x_dot

#         xdot_update = (-2 * self.m_p_l * (
#                     theta_dot ** 2) * s + 3 * self.m_p * self.g * s * c + 4 * action - 4 * self.b * x_dot) / (
#                                   4 * self.total_m - 3 * self.m_p * c ** 2)
#         thetadot_update = (-3 * self.m_p_l * (theta_dot ** 2) * s * c + 6 * self.total_m * self.g * s + 6 * (
#                     action - self.b * x_dot) * c) / (4 * self.l * self.total_m - 3 * self.m_p_l * c ** 2)

#         x = x + x_dot * self.dt
#         x_dot = x_dot + xdot_update * self.dt
#         theta = theta + theta_dot * self.dt
#         theta_dot = theta_dot + thetadot_update * self.dt
        
#         obs = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot]).T
        
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

#         for t in range(self.horizon):
#             action = actions[:, t, :]  # numpy array (batch_size x action dim)
#             if not self.ground_truth:
#                 state_predict = self.model.predict(state, action)
#                 state_next = state_predict + state
#                 state_next[:,5:] = state_predict[:,5:]
#             else:
#                 state_next = self.dumb_step(state, action)
                
#             cost = self.cartpole_cost(state_next, action)  # compute cost
#             costs += cost * self.gamma**t
#             state = copy.deepcopy(state_next)

#         print(np.shape(actions))
            
        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)
            
            if not self.only_prior_model:
                state_predict = self.model.predict(state, action)
                state_next = state_predict + state

                cost = self.ar_cost(state_next, action)  # compute cost
                state = state_next

                if self.prior_safety:
                    state_next_prior = self.dumb_step(state_prior, action)
                    cost += self.ar_cost_prior(state_next_prior, action)
                    state_prior = state_next_prior

                costs += cost * self.gamma**t
            else:
                state_next_prior = self.dumb_step(state_prior, action)
                cost = self.cartpole_cost(state_next_prior, action)
                state_prior = state_next_prior
                costs += cost * self.gamma**t
            
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs
    
    def constraint_violated(self, state):
#         x = state[:, 0]
#         x_dot = state[:, 1]
#         cos_theta = state[:, 2]
#         sin_theta = state[:, 3]
#         theta_dot = state[:, 4]
#         theta_constraint = (cos_theta > 0) & (cos_theta < self.coslimit) & (sin_theta > 0)
#         pos_constraint = (x < -self.x_threshold) | (x > self.x_threshold)
        return False

    
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
#         robot_obs = np.concatenate([spoon_pos-torso_pos, spoon_orient, spoon_pos-self.target_pos, robot_right_joint_positions, head_pos-torso_pos, head_orient, forces]).ravel() 3 3 3 7 3 3 1

        distance_weight = 1.0
        action_weight = 0.01
        food_reward_weight = 1.0
        contact_weight = 100
#         straight_weight = 20
        
#         spoon_orient = state[:,3:6]
        distance_mouth_target = np.linalg.norm(state[:, 6:9], axis=1)
        reward_action = -np.sum(np.square(action), axis=1) # Penalize actions
        
        spilled = (abs(state[:,3] - 1.46) > 0.15) | (abs(state[:,4]) > 0.1)
#         straight_spoon = (abs(state[:,5]) < 0.2)
        reached = distance_mouth_target < 0.03
        reward_food = reached * 20 + spilled * (-100)
        
        contact = state[:,-1] > 0.5
        
        
#         print('costshape', np.shape(distance_mouth_target), np.shape(reward_action), np.shape(reward_food))
        
        
        
        reward = -distance_weight*distance_mouth_target + action_weight*reward_action + food_reward_weight*reward_food - contact_weight*contact
        cost = -reward
        
#         cost = 0
#         if self.env == 'swingup':
#             # mujoco env reward
#             if not obs:
#                 x = state[:, 0]
#                 x_dot = state[:, 1]
#                 theta = state[:, 2]
#                 theta_dot = state[:, 3]
#                 sin_theta = np.sin(theta)
#                 cos_theta = np.cos(theta)
#             else:
#                 # self.add_bound = 0.8
#                 x = state[:, 0]
#                 x_dot = state[:, 1]
#                 cos_theta = state[:, 2]
#                 # todo: initially the GP may predict -1.xxx for cos
#                 # cos_theta[cos_theta < -1] = -1
#                 # cos_theta[cos_theta > 1] = 1
#                 sin_theta = state[:, 3]
#                 theta_dot = state[:, 4]
            
#             action = action.squeeze()

#             length = self.task.l # pole length
#             x_tip_error = x - length*sin_theta
#             y_tip_error = length - length*cos_theta
#             reward = np.exp(-(x_tip_error**2 + y_tip_error**2)/length**2)
            
#             if self.constraint:
#                 violation = self.constraint_violated(state)
#                 reward += violation * self.constraint_reward

#             if self.action_cost:
#                 reward += -0.01 * action**2

#             if self.x_dot_cost:
#                 reward += -0.001 * x_dot**2

#             cost = -reward

#         elif self.env == 'stable':
#             # x [-2.4, 2.4], theta [-0.209, 0.209]
#             # self defined cost
#             x = state[:, 0]
#             x_dot = state[:, 1]
#             theta = state[:, 2]
#             theta_dot = state[:, 3]

#             if env_cost:
#                 # the environment reward, not related to action
#                 done1 = x < -self.task.x_threshold
#                 done2 = x > self.task.x_threshold
#                 done3 = theta < -self.task.theta_threshold_radians
#                 done4 = theta > self.task.theta_threshold_radians
#                 done = np.logical_or(np.logical_or(done1, done2), np.logical_or(done3, done4))
#                 # if done, reward = 1, cost = -1, else reward = 0, cost = 0
#                 cost = -np.ones(done.shape[0]) + done * 1
#             else:
#                 # defined cost
#                 cost = 0.1 * x ** 2 + theta ** 2 + 0.01 * (0.1 * x_dot ** 2 + theta_dot ** 2)
# #                 cost = 0.01 * (x ** 2) - np.cos(theta)
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

