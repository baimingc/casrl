import numpy as np
from .optimizers import RandomOptimizer, CEMOptimizer
import copy

import gym


def create_dynamic_groundtruth(name='Hopperm00-v1'):
    env_gt = gym.make(name)
    return env_gt


class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, mpc_config):
        # mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = np.array(conf["action_low"]) # array (dim,)
        self.action_high = np.array(conf["action_high"]) # array (dim,)
        self.action_dim = conf["action_dim"]
        self.popsize = conf["popsize"]
        self.action_cost = conf["action_cost"]
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
        self.optimizer.setup(self.fetchslide_cost_function)
        self.reset()

        # create a ground truth dynamic
        self.env_gt = create_dynamic_groundtruth()
        self.env_gt.reset()

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).
        Returns: None
        """
        print('set init mean to 0')
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])

    def act(self, task, model, state, ground_truth=False):
        '''
        :param state: task, model, (numpy array) current state
        :return: (float) optimal action
        '''
        self.task = task
        self.model = model
        self.state = state
        self.ground_truth = ground_truth

        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        action = soln[:self.action_dim]

        return action

    def preprocess(self, batch_size=None):
        state = self.state[0:]
        #state = np.concatenate((self.state[0:1], self.state[3:10], self.state[12:18]))
        state = np.repeat(state.reshape(1, -1), self.popsize*self.particle, axis=0)
        return state

    def fetchslide_cost_function(self, actions):
        # the observation need to be processed since we use a common model
        state = self.preprocess()

        # TODO: may be able to change to tensor like pets
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))
        costs = np.zeros(self.popsize*self.particle)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size, timestep, action dim)
            # state_next = self.model.predict(self.index, state, action)+state  # numpy array (batch_size x state dim)
            # the output of the prediction model is [state_next - state]
            if not self.ground_truth:
                state_delta = self.model.predict(state, action) 
                state_next = state_delta + state
                # state_next[0] should be delta x, we dont need to add the original x
                state_next[:, 0] = state_delta[:, 0]
                cost = self.halfcheetah_cost(state_next, action)  # compute cost
            else:
                # change to ground truth one
                # [TODO] For each action, we should run the environment step
                state_next = []
                for i in range(state.shape[0]):
                    self.env_gt.set_model(state[i, :6], state[i, 6:])
                    state_next_i, reward, done, info = self.env_gt.step(action[i])
                    state_next.append(state_next_i)
                state_next = np.array(state_next)
                cost = self.halfcheetah_cost(state_next, action)  # compute cost

            costs += cost * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def halfcheetah_cost(self, state, action):
        heading_penalty_factor = 10
        leg_penalty_factor = 10
        cost = np.zeros((state.shape[0],))

        delta_x = state[:, 0]
        height = state[:, 1] # z
        ang = state[:, 2]    # angle of head
        thigh = state[:, 3]
        leg = state[:, 4]
        foot = state[:, 5]

        # env constraint
        cost[height <= 1.0] += heading_penalty_factor
        cost[abs(ang) >= 0.2] += heading_penalty_factor
        
        cost[thigh <= -0.6] += leg_penalty_factor
        cost[leg <= -0.2] += leg_penalty_factor
        cost[foot <= -0.4] += leg_penalty_factor

        #cost += 3.0*(height - 1.3)**2
        #cost += np.sum(0.1*action**2, axis=1)

        # state[0] is delta_X
        cost -= delta_x/0.04
        return cost
