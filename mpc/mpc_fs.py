import numpy as np
from .optimizers import RandomOptimizer, CEMOptimizer
import copy


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
        self.env = conf["env"]
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

        # FetchSlide requires 4 actions, we need to use 0 as the rest one
        action = np.concatenate((np.array([0.0]), action))
        action = np.concatenate((action, np.array([0.0, 0.0])))
        #action = np.array([0.0, 0.0, 0.0, 0.0])

        return action

    def preprocess(self, batch_size=None):
        observation = self.state['observation']
        desired_goal = self.state['desired_goal'][1:2]

        # according to https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py
        grip_pos = observation[0:2]
        object_pos = observation[3:5]
        object_velp = observation[14:17]
        grip_velp = observation[20:23]

        state = np.concatenate((grip_pos[1:2], grip_velp[1:2], object_pos[1:2], object_velp[1:2]), axis=0)[None]
        state = np.repeat(state.reshape(1, -1), self.popsize*self.particle, axis=0)
        return state, desired_goal

    def fetchslide_cost_function(self, actions):
        # the observation need to be processed since we use a common model
        state, desired_goal = self.preprocess()

        # TODO: may be able to change to tensor like pets
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))
        costs = np.zeros(self.popsize*self.particle)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size, timestep, action dim)
            # state_next = self.model.predict(self.index, state, action)+state  # numpy array (batch_size x state dim)
            # the output of the prediction model is [state_next - state]
            if not self.ground_truth:
                state_next = self.model.predict(state, action) + state
                cost = self.fetchslide_cost(state_next, action, desired_goal)  # compute cost
            else:
                # change to ground truth one
                # [TODO] For each action, we should run the environment step
                state_next = []
                for i in range(state.shape[0]):
                    self.task.set_state(state[i])
                    state_next_i, reward, done, info = self.task.step(action[i])
                    state_next.append(state_next_i)
                state_next = np.array(state_next)
                cost = self.groundtruth_fetchslide_cost(state_next)  # compute cost

            costs += cost * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def fetchslide_cost(self, state, action, desired_goal):
        grip_pos = state[:, 0:1]
        # when the second state is the difference between object and goal
        object_pos = state[:, 2:3]

        vec_1 = object_pos - grip_pos
        vec_2 = object_pos - desired_goal

        # when the object goes too far away, punish it
        overshot_penality = 1.0 * (vec_2[:, 0] < 0)

        cost_near = np.sum(np.abs(vec_1), axis=1)
        cost_dist = np.sum(np.abs(vec_2), axis=1)
        cost = 1.5 * cost_dist + 0.5 * cost_near# + overshot_penality

        if self.action_cost:
            cost_ctrl = np.sum(action**2, axis=1)
            cost += 0.2 * cost_ctrl

        return cost
