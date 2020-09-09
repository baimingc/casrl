from tf.policies.base import Policy
from rllab.core.serializable import Serializable


class UniformControlPolicy(Policy, Serializable):
    def __init__(
            self,
            env,
            *args,
            **kwargs
    ):
        # self.regressors = regressors
        Serializable.quick_init(self, locals())
        super(UniformControlPolicy, self).__init__(env_spec=env)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        return self.action_space.sample(), dict()

    def get_actions(self, observations):
        return self.action_space.sample_n(len(observations)), dict()

    def get_params_internal(self, **tags):
        return None #self.regressors.get_param_values(**tags)
