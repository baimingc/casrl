import copy
import inspect
import pickle
import numpy as np
import itertools
import time
import tensorflow as tf

from rllab.sampler.base import Sampler
from rllab.misc import tensor_utils
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger

from tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from tf.envs.vec_env_executor import VecEnvExecutor

from mole.em_iteration import EM_iteration

class VectorizedSampler(Sampler):

    def __init__(self, env, policy, n_envs, model, vec_env=None, parallel=False, is_maml=False, 
                num_extra=0, tempLR=0.01, temp=0.02, alpha=5, k=10, doing_continual=False, doing_continual_for_onPolicy=False):
        self.env = env
        self.policy = policy
        self.n_envs = n_envs
        self.vec_env = vec_env
        self.env_spec = env.spec
        self.parallel = parallel
        self.model = model
        self.is_maml = is_maml
        self.num_extra = num_extra
        
        self.tempLR = tempLR
        self.temp = temp
        self.alpha = alpha
        self.k = k
        self.doing_continual = doing_continual
        self.doing_continual_for_onPolicy = doing_continual_for_onPolicy

        if(self.is_maml or doing_continual):
            self.update_bs = self.model.config['update_batch_size']
        else:
            self.update_bs=-7
        self.twok = 2*self.k

        if(doing_continual):
            pass
        else:
            self.k = self.update_bs #the chunks that you pass into maml
            self.twok = 2*self.k

        self.list_of_theta_options = []

    def start_worker(self):
        if self.vec_env is None:
            n_envs = self.n_envs
            if getattr(self.env, 'vectorized', False):
                self.vec_env = self.env.vec_env_executor(n_envs=n_envs)
            elif self.parallel:
                self.vec_env = ParallelVecEnvExecutor(
                    env=self.env,
                    n_envs=self.n_envs,
                )
            else:
                #code currently goes into here
                envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_envs)]
                    #even though this pickle call 
                    #will call the env creation function w task=None by default, 
                    #the env.wrapped_env.wrapped_env.task param will indeed be populated correctly
                self.vec_env = VecEnvExecutor(
                    envs=envs,
                )

    def shutdown_worker(self):
        self.vec_env.terminate()

    def update_regressor(self, obs, acts, next_obs, path_length=-7, idx=-7):

        ############################
        ###### NEW SOFT CONTINUOUS WAY
        ############################

        if(self.doing_continual_for_onPolicy):

            junk=[]
            past_probabilities, theta_old, tempLR_junk, _ = EM_iteration(obs, acts, next_obs, path_length, self.model, self.sess, self._weights,
                                self.tempLR_junk_many[idx], junk, self.temp, self.alpha, self.k, self.past_probabilities_many[idx], 
                                self.theta_old_many[idx], continual=self.doing_continual, update_lr=self.model.config['update_lr'])

            self.past_probabilities_many[idx]=past_probabilities
            self.theta_old_many[idx]=theta_old
            self.tempLR_junk_many[idx]=tempLR_junk

        ############################
        ######## OLD WAY
        ############################

        else:
            labels = (next_obs - obs).reshape(1, 2*self.k+self.num_extra, -1)
            inputs = np.concatenate([obs, acts], axis=-1).reshape(1, 2*self.k+self.num_extra, -1)
            if(self.doing_continual):
                feed_dict = {self.model.inputa_kpts: inputs[:,self.k:,:], self.model.labela_kpts: labels[:,self.k:,:]}
            else:
                feed_dict = {self.model.inputa: inputs[:,self.k:,:], self.model.labela: labels[:,self.k:,:]}
            self.update = True

            #restore theta*
            self.model.regressor.set_params(self._weights)

            #use past "update_bs" steps to take a gradient step (or multiple) away from theta*
            for _ in range(self.num_updates):
                self.sess.run([self.model.test_op], feed_dict=feed_dict)


    def obtain_samples(self, max_path_length, batch_size, max_n_trajs=None, seeds=None, show_progress=True, num_updates=1):

        num_extra=self.num_extra
        self.num_updates = num_updates

        #each entry of "paths" corresponds to each env (where each env is a different task)
        paths = {}
        self._weights = self.model.regressor.get_params()

        #add theta* to list of options from which to perform each adaptation step from
        self.list_of_theta_options.append(self._weights)

        self.sess = tf.get_default_session()
        for i in range(self.vec_env.n_envs):
            paths[i] = []

        #init vars
        n_samples = 0
        policy_time = 0
        env_time = 0
        process_time = 0
        steps_taken=0
        dones = np.asarray([True] * self.vec_env.n_envs)
        running_paths = [None] * self.vec_env.n_envs
        if show_progress:
            pbar = ProgBarCounter(batch_size)
        policy = self.policy
        if hasattr(policy, 'inform_vec_env'):
            policy.inform_vec_env(self.vec_env)

        #initial observations, obtained from resetting the envs
        obses = self.vec_env.reset(dones, seeds=seeds)

        #initialize for continual
        if(self.is_maml):
            self.theta_old_many = [[] for i in range(self.vec_env.n_envs)]
            self.tempLR_junk_many = [[] for i in range(self.vec_env.n_envs)]
            self.past_probabilities_many = [[] for i in range(self.vec_env.n_envs)]
            for idx in range(self.vec_env.n_envs):
                self.theta_old_many[idx].append(self._weights)
                self.tempLR_junk_many[idx]= []
                self.tempLR_junk_many[idx].append(self.model.config['update_lr'])
                self.past_probabilities_many[idx].append([])


        #########################################
        ## perform rollouts
        #########################################

        #each rollout = max_path_length
        #do this until total number of points >= batch_size
        while n_samples < batch_size:

            #########################################
            ## get action
            #########################################

            t = time.time()
            policy.reset(dones)
            if policy.is_oneshot:
                _obses = []
                _actions = []
                steps = self.policy.regressor.steps
                for idx in range(len(obses)):
                    _obs = np.zeros((steps, self.env.observation_space.flat_dim))
                    _act = np.zeros((steps, self.env.action_dim))
                    _obs[-1] = obses[idx]
                    if running_paths[idx] is not None:
                        past_obs = running_paths[idx]['observations']
                        past_actions = running_paths[idx]['actions']
                        if len(past_obs) >= steps:
                            _obs[:-1] = np.array(past_obs[-steps+1:])
                            _act = np.array(past_actions[-steps:])
                        else:
                            _obs[-len(past_obs)-1:-1] = np.array(past_obs)
                            _act[-len(past_actions):] = np.array(past_actions)
                    _obses.append(_obs)
                    _actions.append(_act)
                actions, agent_infos = policy.get_actions(_obses, _actions)
            # elif policy.multi_input:
            #     _obses = []
            #     _actions = []
            #     actions = []
            #     steps = policy.multi_input
            #     idxs_ts = np.arange(self.update_bs)
            #     for idx in range(len(obses)):
            #         _obs = np.zeros((steps, self.env.observation_space.flat_dim))
            #         _act = np.zeros((steps, self.env.action_dim))
            #         if running_paths[idx] is not None:
            #             past_obs = running_paths[idx]['observations']
            #             past_actions = running_paths[idx]['actions']
            #             if len(past_obs) >= steps:
            #                 _obs = np.array(past_obs[-steps:])
            #                 _act = np.array(past_actions[-steps:])
            #             else:
            #                 _obs[-len(past_obs):] = np.array(past_obs)
            #                 _act[-len(past_actions):] = np.array(past_actions)
            #         # _obses.append(_obs)
            #         # _actions.append(_act)
            #         if running_paths[idx] is not None and len(running_paths[idx]['observations']) >= self.update_bs + 1 + steps:
            #             obs = np.array(running_paths[idx]['observations'])[-self.update_bs-1:-1]
            #             next_obs = np.array(running_paths[idx]['observations'])[-self.update_bs:]
            #             diff_obs = next_obs - obs
            #             inputs = np.array([np.concatenate(sum([[running_paths[idx]['observations'][t-s],
            #                                             running_paths[idx]['actions'][t-s]]
            #                                             for s in range(policy.multi_input-1,-1, -1)], [])).reshape(-1)
            #                                             for t in idxs_ts])
            #             self.multi_update_regressor(inputs, diff_obs)
            #         action, agent_infos = policy.get_actions([_obs], [_act])
            #         actions.append(action[0])
            #     actions = np.array(actions)
            elif self.is_maml and not self.policy.random:
                actions, agent_infos = [], {}
                currpathlength_list = []
                for idx in range(len(obses)): # for each task (in metaBS)

                    if running_paths[idx] is not None:

                        path_length_real = len(running_paths[idx]['observations'])

                        #reset lists every once in a while (more than what we metatrain on, but not too long)
                        effective_path_length = path_length_real%200
                        if(path_length_real%200==0):
                            self.past_probabilities_many[idx]=[]
                            self.theta_old_many[idx]=[]
                            self.theta_old_many[idx].append(self._weights)
                            self.tempLR_junk_many[idx]=[]
                            self.tempLR_junk_many[idx].append(self.model.config['update_lr'])

                        if effective_path_length> (self.twok + num_extra):
                            obs = np.array(running_paths[idx]['observations'])[-self.twok-num_extra-1:-1]
                            acts = np.array(running_paths[idx]['actions'])[-self.twok-num_extra-1:-1]
                            next_obs = np.array(running_paths[idx]['observations'])[-self.twok-num_extra:]

                            self.update_regressor(obs, acts, next_obs, effective_path_length, idx) 
                            action, agent_info = policy.get_action(obses[idx])

                        else:
                            action, agent_infos = policy.get_action(obses[idx])
                    else:
                        path_length_real=0
                        action, agent_infos = policy.get_action(obses[idx])
                    actions.append(action[0])
                    currpathlength_list.append(path_length_real)
                actions = np.array(actions)
                currpathlength_list=np.array(currpathlength_list)
            else:
                ######################### TO DO: remove this
                '''for idx in range(len(obses)): # for each task (in metaBS)
                    if running_paths[idx] is not None:
                        path_length_real = len(running_paths[idx]['observations'])
                        print(path_length_real, "... ", max_path_length, " ... task index: ", idx)'''
                #########################

                currpathlength_list=np.zeros((len(obses),))
                actions, agent_infos = policy.get_actions(obses)
            policy_time += time.time() - t
            
            #########################################
            ## take 1 step
            #########################################

            t = time.time()
            
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions, max_path_length=max_path_length, currpathlength_list=currpathlength_list)
            
            ############TO DO: remove this
            '''if(np.any(currpathlength_list>0)):
                print("CURR PATH LENGTHS: ", currpathlength_list)
                if(np.any(dones)):
                    print("DONES: ", dones)'''
            ###########################

            if np.any(dones):
                new_obses = self.vec_env.reset(dones)
                reset_idx = 0
                for idx, done in enumerate(dones):
                    if done:
                        next_obses[idx] = new_obses[reset_idx]
                        reset_idx += 1
            env_time += time.time() - t

            #########################################
            ## append the info from that step into "running_paths"
            #########################################

            #get env and agent infos
            t = time.time()
            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.n_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.n_envs)]

            #iterate through each env (where each env is a diff task)
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):

                #on the first iter of this while loop, running_paths is empty, so initialize it (for each env)
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )

                #during each iteration of the while loop, add onto observations/etc. (for each env)
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                #if that env is "done", copy the info from running_paths and append it onto paths
                if done:
                    paths[idx].append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))

                    #total number of datapoints collected (across all envs)
                    n_samples += len(running_paths[idx]["rewards"])

                    #reset "running_paths" back to empty, for that env
                    running_paths[idx] = None

                    #reset lists for continual
                    if(self.is_maml):
                        self.past_probabilities_many[idx]=[]
                        self.theta_old_many[idx]=[]
                        self.theta_old_many[idx].append(self._weights)
                        self.tempLR_junk_many[idx]=[]
                        self.tempLR_junk_many[idx].append(self.model.config['update_lr'])

            #check if should stop performing rollouts
            if max_n_trajs is not None and len(paths) >= max_n_trajs:
                break

            #some bookkeeping
            steps_taken += 1
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()
        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)
        if self.is_maml and not policy.random:
            self.model.regressor.set_params(self._weights)
        return paths