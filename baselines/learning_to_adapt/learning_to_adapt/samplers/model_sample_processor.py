'''
@Author: 
@Email: 
@Date: 2020-04-06 20:40:15
@LastEditTime: 2020-04-06 22:19:03
@Description: 
'''

from learning_to_adapt.samplers.base import SampleProcessor
from learning_to_adapt.utils import tensor_utils
import numpy as np


class ModelSampleProcessor(SampleProcessor):
    def __init__(
            self,
            baseline=None,
            discount=0.99,
            gae_lambda=1,
            normalize_adv=False,
            positive_adv=False,
            recurrent=False
    ):

        self.baseline = baseline
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv
        self.recurrent = recurrent

    def process_samples(self, paths, log=False, log_prefix=''):
        """ Compared with the standard Sampler, ModelBaseSampler.process_samples provides 3 additional data fields
                - observations_dynamics
                - next_observations_dynamics
                - actions_dynamics
            since the dynamics model needs (obs, act, next_obs) for training, observations_dynamics and actions_dynamics
            skip the last step of a path while next_observations_dynamics skips the first step of a path
        """

        assert len(paths) > 0
        recurrent = self.recurrent
        # compute discounted rewards - > returns
        returns = []
        for idx, path in enumerate(paths):
            path["returns"] = tensor_utils.discount_cumsum(path["rewards"], self.discount)
            returns.append(path["returns"])


        # 8) log statistics if desired
        self._log_path_stats(paths, log=log, log_prefix=log_prefix)

        observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][:-1] for path in paths], recurrent)
        next_observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][1:] for path in paths], recurrent)
        actions_dynamics = tensor_utils.concat_tensor_list([path["actions"][:-1] for path in paths], recurrent)
        timesteps_dynamics = tensor_utils.concat_tensor_list([np.arange((len(path["observations"]) - 1)) for path in paths])

        rewards = tensor_utils.concat_tensor_list([path["rewards"][:-1] for path in paths], recurrent)
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths], recurrent)

        samples_data = dict(
            observations=observations_dynamics,
            next_observations=next_observations_dynamics,
            actions=actions_dynamics,
            timesteps=timesteps_dynamics,
            rewards=rewards,
            returns=returns,
        )

        return samples_data
