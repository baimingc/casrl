from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
import tensorflow as tf
from distutils.version import LooseVersion


def worker_init_tf(G):
    if not hasattr(G, 'sess') or G.sess is None:
        G.sess = tf.Session()
        G.sess.__enter__()


def worker_init_tf_vars(G):
    if LooseVersion(tf.__version__) >= '0.12.1':
        # this suppresses annoying warning messages from tf
        initializer = tf.global_variables_initializer
    else:
        initializer = tf.initialize_all_variables
    G.sess.run(initializer())


class BatchSampler(BaseSampler):
    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.algo.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr, max_path_length, batch_size):
        cur_policy_params = self.algo.policy.get_param_values()
        if hasattr(self.algo.env,"get_param_values"):
            cur_env_params = self.algo.env.get_param_values()
        else:
            cur_env_params = None
        paths = parallel_sampler.sample_paths(
            policy_params=cur_policy_params,
            env_params=cur_env_params,
            max_samples=batch_size,
            max_path_length=max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated