import numpy as np

from tf.core.layers_powered import LayersPowered
import tf.core.layers as L
from tf.core.network import MLP
from tf.spaces.box import Box
from tf.misc import tensor_utils
from tf.policies.base import StochasticPolicy
from tf.distributions.diagonal_gaussian import DiagonalGaussian

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger

import tensorflow as tf

from sandbox.ignasi.optimizers.first_order_optimizer import FirstOrderOptimizer


class GaussianMLPPolicy(StochasticPolicy, LayersPowered, Serializable):
    @property
    def vectorized(self):
        return True

    def __init__(
            self,
            env,
            regressor,
            name='policy',
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=0.5,
            adaptive_std=True,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            mean_network=None,
            std_network=None,
            std_parametrization='exp',
            normalize_inputs=True,
            normalize_outputs=True,
            optimizer=None,
            optimizer_args=dict(max_epochs=100, verbose=False),
            n_candidates=1000,
            horizon=10,
            *args,
            **kwargs
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """
        env_spec = env.spec
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        with tf.variable_scope(name):

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            if optimizer is None:
                optimizer = FirstOrderOptimizer(**optimizer_args)
            elif optimizer is 'maml':
                raise NotImplementedError

            self._optimizer = optimizer
            self._name = name
            self._normalize_inputs = normalize_inputs
            self._normalize_outputs = normalize_outputs
            self._n_candidates = n_candidates
            self._horizon = horizon
            self.regressor = regressor
            self.get_reward = env._wrapped_env._wrapped_env.get_reward
            self.random = False
            self.nn_policy = False
            self.is_oneshot = False

            # create network
            if mean_network is None:
                mean_network = MLP(
                    name="mean_network",
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                )
            self._mean_network = mean_network

            l_mean = mean_network.output_layer
            obs_var = mean_network.input_layer.input_var

            self.log_std = np.log(init_std)
            self.log_std_var = tf.Variable(
                self.log_std,
                name="log_std",
                trainable=False,
            )


            # mean_var, log_std_var = L.get_output([l_mean, l_std_param])
            #
            # if self.min_std_param is not None:
            #     log_std_var = tf.maximum(log_std_var, np.log(min_std))
            #
            # self._mean_var, self._log_std_var = mean_var, log_std_var

            self._l_mean = l_mean

            self._dist = DiagonalGaussian(action_dim)

            LayersPowered.__init__(self, [l_mean])
            super(GaussianMLPPolicy, self).__init__(env_spec)

            dist_info_sym = self.dist_info_sym(mean_network.input_layer.input_var, dict())
            mean_var = dist_info_sym["mean"]

            self._f_dist = tensor_utils.compile_function(
                inputs=[obs_var],
                outputs=[mean_var],
            )

            xs_var = mean_network.input_layer.input_var
            ys_var = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name="ys")
            self.xs_var = xs_var

            x_mean_var = tf.get_variable(
                name="x_mean",
                shape=(1,obs_dim),
                initializer=tf.constant_initializer(0., dtype=tf.float32),
                trainable=False,
            )
            x_std_var = tf.get_variable(
                name="x_std",
                shape=(1,obs_dim),
                initializer=tf.constant_initializer(1., dtype=tf.float32),
                trainable=False
            )

            y_mean_var = tf.Variable(
                np.zeros((1, action_dim), dtype=np.float32),
                name="y_mean",
                trainable=False,
            )
            y_std_var = tf.Variable(
                np.ones((1, action_dim), dtype=np.float32),
                name="y_std",
                trainable=False,
            )

            normalized_xs_var = (xs_var - x_mean_var) / x_std_var
            normalized_ys_var = (ys_var - y_mean_var) / y_std_var
            self.normalized_xs_var = normalized_xs_var

            fit_ys_var = L.get_output(l_mean, {mean_network.input_layer: normalized_xs_var})
            self.fit_ys_var = fit_ys_var

            loss = 0.5 * tf.reduce_mean(tf.square(fit_ys_var - normalized_ys_var))

            self.f_predict = tensor_utils.compile_function([xs_var], fit_ys_var)

            optimizer_args = dict(
                loss=loss,
                target=self,
                network_outputs=[fit_ys_var],
            )

            optimizer_args["inputs"] = [xs_var, ys_var]

            self._optimizer.update_opt(**optimizer_args)

            self.name = name
            self.l_mean = l_mean

            self.normalize_inputs = normalize_inputs
            self._x_mean_var = x_mean_var
            self._x_std_var = x_std_var
            self._y_mean_var = y_mean_var
            self._y_std_var = y_std_var
            self._x_mean, self._y_mean = 0, 0
            self._x_std, self._y_std = 1, 1

            unnorm_fit_ys_var = fit_ys_var * y_std_var + y_mean_var

    def fit(self, xs, ys):
        sess = tf.get_default_session()
        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            new_mean = np.mean(xs, axis=0, keepdims=True)
            new_std = np.std(xs, axis=0, keepdims=True) + 1e-8
            tf.get_default_session().run(tf.group(
                tf.assign(self._x_mean_var, new_mean),
                tf.assign(self._x_std_var, new_std),
            ))
            self._x_mean, self._x_std = sess.run([self._x_mean_var, self._x_std_var])
        if self._normalize_outputs:
            # recompute normalizing constants for outputs
            sess.run([
                tf.assign(self._y_mean_var, np.mean(ys, axis=0, keepdims=True)),
                tf.assign(self._y_std_var, np.std(ys, axis=0, keepdims=True) + 1e-8),
            ])
            self._y_mean, self._y_std = sess.run([self._y_mean_var, self._y_std_var])
        inputs = [xs, ys]
        loss_before = self._optimizer.loss(inputs)
        if self._name:
            prefix = self._name + "_"
        else:
            prefix = ""
        logger.record_tabular(prefix + 'LossBefore', loss_before)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        logger.record_tabular(prefix + 'LossAfter', loss_after)
        logger.record_tabular(prefix + 'dLoss', loss_before - loss_after)

    def dist_info_sym(self, obs_var, state_info_vars=None):
        mean_var = L.get_output([self._l_mean], obs_var)
        return dict(mean=mean_var)

    # This function is just used for testing, so there is no need to include the random action
    @overrides
    def get_action(self, observation):
        if type(observation) is not list:
            observation = [observation]
        if self.random:
            return self.get_random_action(1), dict()
        else:
            if self.nn_policy:
                return self._get_action(observation)
            else:
                actions = self.get_best_action(observation)
                return actions, dict()

        # visualize regressor's multi-step prediction
        # calls function inside it to get best action
        # if self._test_regressor:
        #     return self.get_prediction(observation), dict()

    #################################################################

    # Get Action: either best (according to MPC) or random
    @overrides
    def get_actions(self, observations):
        if(self.random):
            return self.get_random_action(len(observations)), dict()
        else:
            if self.nn_policy:
                return self._get_actions(observations)
            else:
                actions = self.get_best_action(observations)
                return actions, dict()

    def get_random_action(self, n):
        return self.action_space.sample_n(n)

    def get_best_action(self, observation):

        # init
        n = self._n_candidates
        m = len(observation)
        R = np.zeros((n * m,))

        # randomly sample n sequences of length-h actions (for multiple envs in parallel)
        a = self.get_random_action(self._horizon * n * m).reshape((self._horizon, n * m, -1))

        # simulate the action sequences
        for h in range(self._horizon):
            if h == 0:
                cand_a = a[h].reshape((m, n, -1))
                observation = np.array(observation).reshape((m, 1, -1))
                observation = np.tile(observation, (1, n, 1)).reshape(
                    (m * n, -1))  # The observations are ob_1,...,ob_1, ob_2, ..., ob_2,....
            next_observation = self.regressor.predict(np.concatenate([observation, a[h]], axis=1)) + observation
            r = self.get_reward(observation, next_observation, a[h])
            R += r
            observation = next_observation
        R = R.reshape(m, n)

        # return the action from the sequence that resulted in highest reward
        return cand_a[range(m), np.argmax(R, axis=1)]

    def _get_action(self, observation):
        observation = observation[0]
        if self._normalize_inputs:
            observation = (observation - self._x_mean)/self._x_std
        flat_obs = self.observation_space.flatten(observation)
        mean = [x[0] for x in self._f_dist([flat_obs])][0]
        log_std = self.log_std
        if self._normalize_outputs:
            mean = mean * self._y_std + self._y_mean
        rnd = np.random.normal(size=mean.shape)
        action = mean
        return action, dict(mean=mean, log_std=log_std)

    def _get_actions(self, observations):
        if self._normalize_inputs:
            observations = (observations - self._x_mean)/self._x_std
        flat_obs = self.observation_space.flatten_n(observations)
        means = self._f_dist(flat_obs)
        log_stds = [self.log_std] * len(means)
        if self._normalize_outputs:
            means = means * self._y_std + self._y_mean
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    @property
    def distribution(self):
        return self._dist
