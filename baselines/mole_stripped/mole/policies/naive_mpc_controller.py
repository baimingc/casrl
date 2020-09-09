from tf.policies.base import Policy
from rllab.core.serializable import Serializable
import numpy as np
from mole.utils import PlotRegressor
import tensorflow as tf
import time
import matplotlib

# matplotlib.use('Agg')

class NaiveMPCController(Policy, Serializable):
    def __init__(
            self,
            env,
            regressor,
            n_candidates=1000,
            horizon=10,
            test_regressor=False,
    ):
        self.counter = 0
        self.obs = []
        self.pred_obs = []
        self.actions = []
        self.env = env
        self.get_reward = env._wrapped_env._wrapped_env.get_reward
        self.regressor = regressor
        self.is_oneshot = regressor.is_oneshot
        self.is_rnn = regressor.is_rnn
        self._n_candidates = n_candidates
        self._horizon = horizon
        self.random = False
        self._test_regressor = test_regressor
        self.init_state = None
        self.first = True
        self.multi_input = regressor.multi_input
        if test_regressor:
            self.plot = PlotRegressor(env._wrapped_env._wrapped_env.__class__.__name__)
            #NOTE: setting this to True doesnt give you an optimal rollout
                #it plans only evey horizon-steps, and doesnt replan at each step
                    #because it's trying to verify accuracy of regressor
        self.i = 0
        Serializable.quick_init(self, locals())
        super(NaiveMPCController, self).__init__(env_spec=env.spec)

    @property
    def vectorized(self):
        return True

    #################################################################3

    #This function is just used for testing, so there is no need to include the random action
    def get_action(self, observation):
        if type(observation) is not list:
            observation =[observation]
        
        #visualize regressor's multi-step prediction 
        #calls function inside it to get best action
        if self._test_regressor:
            return self.get_prediction(observation), dict()
        
        #get best action
        action = self.get_best_action(observation)
        return action, dict()

    #################################################################
    
    #Get Action: either best (according to MPC) or random
    def get_actions(self, observations, actions=None):
        if self.random:
            actions = self.get_random_action(len(observations))
        else:
            actions = self.get_best_action(observations, actions)
        return actions, dict()

    def get_random_action(self, n):
        return self.action_space.sample_n(n)

    def _get_best_action_rnn(self, observation):
        # init
        n = self._n_candidates
        m = len(observation)
        R = np.zeros((n * m,))
        next_state = self.init_state

        # randomly sample n sequences of length-h actions (for multiple envs in parallel)
        a = self.get_random_action(self._horizon * n * m).reshape((self._horizon, n * m, -1))

        # simulate the action sequences
        for h in range(self._horizon):
            if h == 0:
                cand_a = a[h].reshape((m, n, -1))
                observation = np.array(observation).reshape((m, 1, -1))
                observation = np.tile(observation, (1, n, 1)).reshape(
                    (m * n, -1))  # The observations are ob_1,...,ob_1, ob_2, ..., ob_2,....
            next_observation, next_state = self.regressor.predict(np.concatenate([observation, a[h]], axis=1),
                                                                      next_state, first=self.first)
            next_observation += observation
            if h == 0:
                cand_states = [(s.c.reshape((m, n, -1)), s.h.reshape((m, n, -1))) for s in next_state]
                self.first = False
            r = self.get_reward(observation, next_observation, a[h])
            R += r
            observation = next_observation
        R = R.reshape(m, n)
        self.init_state = tuple(
                [(c[range(m), np.argmax(R, axis=1)], h[range(m), np.argmax(R, axis=1)]) for c, h in cand_states])
        self.init_state = tuple([(
                                    np.tile(c, (1, 1, n)).reshape(m * n, -1),
                                    np.tile(h, (1, 1, n)).reshape(m * n, -1))
                                    for c, h in self.init_state]
                                    )
        # return the action from the sequence that resulted in highest reward
        return cand_a[range(m), np.argmax(R, axis=1)]

    def _get_best_action_oneshot(self, observation, actions):
        n = self._n_candidates
        m = len(observation)
        R = np.zeros((n * m,))
        # I need to pass the actions!!!

        # randomly sample n sequences of length-h actions (for multiple envs in parallel)
        random_act = self.get_random_action(self._horizon * n * m).reshape((self._horizon, n * m, -1))
        act = actions

        # simulate the action sequences

        for h in range(self._horizon):
            if h == 0:
                cand_a = random_act[h].reshape((m, n, -1))
                observation = np.array(observation).reshape((self.regressor.steps, m, 1, -1))
                observation = np.tile(observation, (1, 1, n, 1)).reshape(
                    (self.regressor.steps, m * n, -1))  # The observations are ob_1,...,ob_1, ob_2, ..., ob_2,....
                next_observation = observation[-1]
                act = np.array(act).reshape((self.regressor.steps, m, 1, -1))
                act = np.tile(act, (1, 1, n, 1)).reshape(
                    (self.regressor.steps, m * n, -1))  # The observations are ob_1,...,ob_1, ob_2, ..., ob_2,....
            act = np.concatenate([act, np.expand_dims(random_act[h], 0)])[-self.regressor.steps:]
            diff_observation = self.regressor.predict(np.concatenate([observation, act], axis=-1))
            next_observation += diff_observation
            r = self.get_reward(observation[-1], next_observation, random_act[h])
            R += r
            observation = np.concatenate([observation, np.expand_dims(next_observation, 0)], axis=0)[-self.regressor.steps:]
        R = R.reshape(m, n)
        return cand_a[range(m), np.argmax(R, axis=1)]

    def _get_best_action_multi_input(self, observation, actions):
        n = self._n_candidates
        m = len(observation)
        R = np.zeros((n * m,))

        # randomly sample n sequences of length-h actions (for multiple envs in parallel)
        random_act = self.get_random_action(self._horizon * n * m).reshape((self._horizon, n * m, -1))
        act = actions

        for h in range(self._horizon):
            if h == 0:
                # import pdb; pdb.set_trace()
                cand_a = random_act[h].reshape((m, n, -1))
                observation = np.array(observation).reshape((self.multi_input, m, 1, -1))
                observation = np.tile(observation, (1, 1, n, 1)).reshape((self.multi_input, m * n, -1))  # The observations are ob_1,...,ob_1, ob_2, ..., ob_2,....
                next_observation = observation[-1]
                act = np.array(act).reshape((self.multi_input, m, 1, -1))
                act = np.tile(act, (1, 1, n, 1)).reshape(
                    (self.multi_input, m * n, -1))
            act = np.concatenate([act, np.expand_dims(random_act[h], 0)])[-self.multi_input:]
            diff_observation = self.regressor.predict(np.concatenate(sum([[observation[i], act[i]] for i in range(self.multi_input)], []), axis=-1).reshape(m*n, -1))
            next_observation += diff_observation
            r = self.get_reward(observation[-1], next_observation, random_act[h])
            R += r
            observation = np.concatenate([observation, np.expand_dims(next_observation, 0)], axis=0)[-self.multi_input:]
        R = R.reshape(m, n)
        return cand_a[range(m), np.argmax(R, axis=1)]

    def get_best_action(self, observation, actions=None):
        if self.is_oneshot:
            return self._get_best_action_oneshot(observation, actions)
        elif self.is_rnn:
            return self._get_best_action_rnn(observation)
        elif self.multi_input:
            return self._get_best_action_multi_input(observation, actions)
        else:
            #init
            n = self._n_candidates
            m = len(observation)
            R = np.zeros((n*m,))
            next_state = self.init_state

            #randomly sample n sequences of length-h actions (for multiple envs in parallel)
            a = self.get_random_action(self._horizon*n*m).reshape((self._horizon, n*m, -1))

            #simulate the action sequences
            for h in range(self._horizon):
                if h == 0:
                    cand_a = a[h].reshape((m,n,-1))
                    observation = np.array(observation).reshape((m,1,-1))
                    observation = np.tile(observation, (1,n,1)).reshape((m*n, -1)) # The observations are ob_1,...,ob_1, ob_2, ..., ob_2,....
                next_observation = self.regressor.predict(np.concatenate([observation, a[h]], axis=1)) + observation
                r = self.get_reward(observation, next_observation, a[h])
                R += r
                observation = next_observation
            R = R.reshape(m,n)
            #return the action from the sequence that resulted in highest reward
            return cand_a[range(m), np.argmax(R, axis=1)]

    #################################################################

    def get_params_internal(self, **tags):
        return []

    def get_prediction(self, observation):
        self.obs.append(observation[0])

        #at the beginning, populate the pred_obs list with multi-step predictions
        if self.counter == 0:

            #start with current observation
            self.pred_obs.append(observation[0])

            #for each timestep
            for h in range(self._horizon):

                #get current predicted observation
                obs = self.pred_obs[h].copy()

                #get best action
                self.actions.append(self.get_best_action(observation)[0])
                a = self.actions[h].copy()

                #get next predicted observation
                next_observation = self.regressor.predict([np.concatenate([obs, a])])[0] + obs
                self.pred_obs.append(next_observation)
                self.counter = 1
            return self.actions[0]

        #after h timesteps, you have enough in self.obs, so plot the differences
        elif self.counter == self._horizon:
            self.counter = 0
            self.plot.plot(self.obs, self.pred_obs, self.regressor._y_std_var)
            self.obs = []
            self.pred_obs = []
            self.actions = []
            ###return self.get_random_action(1)
            return self.get_best_action(observation)

        #for every timestep, execute the actions that you initially planned
            #to see difference between predicted and true observations
        else:
            act = self.actions[self.counter]
            self.counter += 1
            return act

    def reset(self, dones=None):
        if dones is not None and dones.any():
            self.first = True

