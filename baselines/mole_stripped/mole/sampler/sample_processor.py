import numpy as np

from rllab.algos import util
from rllab.misc import special
from rllab.misc import logger
from tf.misc import tensor_utils


class DefaultSampleProcessor(object):

    def __init__(self, algo):
        self.algo = algo

    def process_samples(self, itr, paths, not_doing_mbrl=False, which_timeseries='household'):

        if(not_doing_mbrl):

            if(type(paths)==np.ndarray):
                samples_data = [self._process_samples(itr, mbs, not_doing_mbrl=not_doing_mbrl, which_timeseries=which_timeseries) for mbs in paths]
            else:
                samples_data = self._process_samples(itr, paths, is_dict=False, not_doing_mbrl=not_doing_mbrl, which_timeseries=which_timeseries)
        else:

            if type(paths) is dict:
                samples_data = [self._process_samples(itr, task) for task in paths.values()]
            elif type(paths) is list:
                samples_data = self._process_samples(itr, paths, is_dict=False)
            else:
                raise TypeError
        return samples_data

    def _process_samples(self, itr, paths, is_dict=True, not_doing_mbrl=False, which_timeseries='household'):

        if(not_doing_mbrl):
            if(is_dict):
                if(which_timeseries=='household' or which_timeseries=='sin'):
                    observations = [np.expand_dims(path["observations"][:-1],1) for path in paths]
                    diff_observations = [np.expand_dims(path["observations"][1:] - path["observations"][:-1], 1) for path in paths]
                else:
                    observations = [path["observations"][:-1] for path in paths]
                    diff_observations = [(path["observations"][1:] - path["observations"][:-1]) for path in paths]
                actions = [path['actions'][:-1] for path in paths]
                rewards = [path["rewards"] for path in paths]
                obs_act = [np.concatenate([observations[i], actions[i]], axis=1) for i in range(len(observations))]
                returns=0
                undiscounted_returns=0
            else:
                observations = tensor_utils.concat_tensor_list([np.concatenate(path["observations"]) for path in paths])
                diff_observations = tensor_utils.concat_tensor_list([np.concatenate(path["diff_observations"]) for path in paths])
                actions = tensor_utils.concat_tensor_list([np.concatenate(path['actions']) for path in paths])
                rewards = 0
                returns=0
                undiscounted_returns=0
                obs_act = tensor_utils.concat_tensor_list([np.concatenate(path["obs_act"]) for path in paths])
        else:
            if is_dict:
                observations = [path["observations"][:-1] for path in paths]
                diff_observations = [path["observations"][1:] - path["observations"][:-1] for path in paths]
                actions = [path['actions'][:-1] for path in paths]
                rewards = [path["rewards"] for path in paths]
                obs_act = [np.concatenate([observations[i], actions[i]], axis=1) for i in range(len(observations))]

            else:
                observations = tensor_utils.concat_tensor_list([np.concatenate(path["observations"]) for path in paths])
                diff_observations = tensor_utils.concat_tensor_list([np.concatenate(path["diff_observations"]) for path in paths])
                actions = tensor_utils.concat_tensor_list([np.concatenate(path['actions']) for path in paths])
                rewards = tensor_utils.concat_tensor_list([np.concatenate(path["rewards"]) for path in paths])
                obs_act = tensor_utils.concat_tensor_list([np.concatenate(path["obs_act"]) for path in paths])

            if 'returns' not in paths[0].keys():
                returns = []
                for idx, path in enumerate(paths):
                    path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
                    returns.append(path["returns"])
                returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
                undiscounted_returns = [sum(path["rewards"]) for path in paths]
            else:
                returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
                undiscounted_returns = np.array(sum([path['undiscounted_returns'] for path in paths], []))
                logger.record_tabular('Iteration', itr)
                logger.record_tabular('NumTasks', len(paths))
                logger.record_tabular_misc_stat('Return', undiscounted_returns, placement='front')
                logger.record_tabular_misc_stat('Action', actions)

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            undiscounted_returns=undiscounted_returns,
            diff_observations=diff_observations,
            obs_act=obs_act,
        )


        return samples_data
