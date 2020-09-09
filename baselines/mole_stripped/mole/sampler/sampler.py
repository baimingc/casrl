import time

from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
# import rllab.plotter as plotter
import tensorflow as tf
import numpy as np

from mole.sampler.sample_processor import DefaultSampleProcessor
from mole.sampler.vectorized_sampler import VectorizedSampler

from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
from distutils.version import LooseVersion

from tf.policies.base import Policy
from tf.misc import tensor_utils


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
        if hasattr(self.algo.env, "get_param_values"):
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




class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
          self,
          env,
          policy,
          regressor,
          model,
          batch_size=50000,
          batch_size_initial=50000,
          max_buffer=1000000,
          meta_bs=64,
          update_bs=10,
          max_epochs=80,
          scope=None,
          n_itr=5,
          n_itr_rand=5,
          start_itr=0,
          max_path_length=333,
          discount=0.99,
          gae_lambda=1,
          plot=False,
          pause_for_plot=False,
          center_adv=True,
          positive_adv=False,
          store_paths=False,
          whole_paths=True,
          sampler=None,
          sample_processor_cls=None,
          sample_processor_args=None,
          n_vectorized_envs=None,
          parallel_vec_env=False,
          correlated_inputs=True,
          train_policy=False,
          n_itr_policy=10,
          rnn=False,
          num_extra=0,
          oneshot=False,
          maml=False,
          multi_input=0,
          tempLR=0.01,
          temp=0.02,
          alpha=5,
          k=10,
          doing_continual=False,
          doing_continual_for_onPolicy=False,
          not_doing_mbrl=False,
          time_series_data=0,
          which_timeseries='household',
          **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        """
        self.env = env
        self.policy = policy
        self.regressor = regressor
        self.max_epochs = max_epochs
        self.scope = scope
        self.n_itr_rand = n_itr_rand
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.batch_size = batch_size * meta_bs
        self.batch_size_initial = batch_size_initial * meta_bs
        self.update_bs = update_bs
        self.meta_bs = meta_bs
        self.itr = 0
        self._y_mean, self._x_mean = 0, 0
        self._y_std, self._y_std = 1, 1
        self._counter = 0
        self._epochs = 0
        self._data = None
        self.n_data_points = 0
        self.max_tasks = int(np.ceil(max_buffer / batch_size))
        self.correlated_inputs = correlated_inputs
        self.rnn = rnn
        self.oneshot = oneshot
        self.multi_input = multi_input
        self.not_doing_mbrl = not_doing_mbrl
        self.time_series_data = time_series_data
        self.which_timeseries=which_timeseries
        if rnn:
            self.update_bs = self.max_path_length

        print("BATCH SIZE: ", batch_size)
        print("BATCH SIZE INITIAL: ", batch_size_initial)

        self.train_policy = train_policy
        self.n_itr_policy = n_itr_policy
        self._dict_data = {}
        assert multi_input <= self.update_bs

        if sampler is None:
            if n_vectorized_envs is None:
                n_vectorized_envs = self.meta_bs
            if self.policy.vectorized:
                sampler = VectorizedSampler(env=env, policy=policy, n_envs=n_vectorized_envs, 
                                        model=model, parallel=parallel_vec_env, is_maml=maml, num_extra=num_extra,
                                        tempLR=tempLR, temp=temp, alpha=alpha, k=k, 
                                        doing_continual=doing_continual, doing_continual_for_onPolicy=doing_continual_for_onPolicy)
            else:
                sampler = BatchSampler(self)

        self.sampler = sampler

        if sample_processor_cls is None:
            sample_processor_cls = DefaultSampleProcessor
        if sample_processor_args is None:
            sample_processor_args = dict(algo=self)
        else:
            sample_processor_args = dict(sample_processor_args, algo=self)

        self.sample_processor = sample_processor_cls(**sample_processor_args)
        # tensor_utils.initialize_new_variables(sess=self.sess)

        
    def start_worker(self):
        self.sampler.start_worker()
        # if self.plot:
        #     plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    #perform rollouts
        #this returns list of length = meta_bs (# envs/tasks)
        #each entry is of length = number of rollouts performed for that task
            #each of those has 5 entries (observations, etc.)
                #each of those entries is of length = rollout length (either max_path length, or length until "done" happened)
    def obtain_samples(self, itr, firstTime=False, num_updates=1):

        if firstTime:
            use_batchsize = self.batch_size_initial
        else:
            use_batchsize = self.batch_size

        ####### paths is:
                #list of length = mbs
                #each entry = list of rollouts
                #each rollout = dict of ['observations', 'actions', 'agent_infos', 'rewards', 'env_infos']
        self.sampling_freq = 1
        if(self.not_doing_mbrl):

            need_to_save_data= True ## False #####False
            if(firstTime):

                if(self.which_timeseries=='household'):

                    self.sampling_freq = 10

                    if(need_to_save_data):

                        ## data: ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
                        ## output: index [2] (of next timestep) - dim 1
                        ## inputs: index [2:] (of curr timestep) - dim 7 (make this s =1, a=6)

                        all_dates = np.copy(self.time_series_data.Date) #shape(2075259,)
                        all_times = np.copy(self.time_series_data.Time)
                        all_states = np.copy(self.time_series_data.Global_active_power) 
                        all_a0 = np.copy(self.time_series_data.Global_reactive_power) 
                        all_a1 = np.copy(self.time_series_data.Voltage) 
                        all_a2 = np.copy(self.time_series_data.Global_intensity) 
                        all_a3 = np.copy(self.time_series_data.Sub_metering_1) 
                        all_a4 = np.copy(self.time_series_data.Sub_metering_2) 
                        all_a5 = np.copy(self.time_series_data.Sub_metering_3) 

                        #make each "task" be a different month of a different year
                        self.timeseries_holdout_data = []
                        self.timeseries_training_data = []
                        
                        prev_month = all_dates[0].split('/')[1]
                        s_thistask = []
                        a_thistask = []

                        #500k points gets you almost through the whole year
                        for i in range(all_dates.shape[0]):
                            #################for i in range(600000):
                            curr_month = all_dates[i].split('/')[1]
                            list_length_limit = 2000

                            #check for nan
                            got_a_nan = False
                            if(np.isnan(all_states[i]) or np.isnan(all_a0[i]) or np.isnan(all_a1[i])
                                or np.isnan(all_a2[i]) or np.isnan(all_a3[i]) or np.isnan(all_a4[i]) or np.isnan(all_a5[i])):
                                got_a_nan=True
                                print("****** got nan at index: ", i, " ... for month ", curr_month)

                            #IF NEW MONTH, OR GOT A NAN, OR LONGER THAN A LIMIT, add as a new task entry
                            if(not(curr_month==prev_month) or (got_a_nan) or (len(s_thistask)>list_length_limit)):
                                print("looking at month: ", curr_month, " ..... data point ", i, "/", all_dates.shape[0])

                                

                                if(len(s_thistask)>2*self.sampling_freq*self.update_bs):

                                    #done collecting all data for this month, so turn each thing into array and make a dict
                                    this_task_dict = dict(observations=np.array(s_thistask)[::self.sampling_freq], actions=np.array(a_thistask)[::self.sampling_freq], 
                                                        agent_infos=0, rewards=0, env_infos=0, month=curr_month)
                                    
                                    #add this to either training or validation, depending on the month
                                    if(curr_month=='12' or curr_month=='6'): #june and december
                                        self.timeseries_holdout_data.append([this_task_dict])
                                    else:
                                        self.timeseries_training_data.append([this_task_dict])

                                #update what month you're on
                                prev_month = curr_month
                                s_thistask = []
                                a_thistask = []
                                if(not(got_a_nan)):
                                    s_thistask.append(all_states[i])
                                    a_thistask.append(np.array([all_a0[i], all_a1[i], all_a2[i], all_a3[i], all_a4[i], all_a5[i]]))

                            else: 
                                s_thistask.append(all_states[i])
                                a_thistask.append(np.array([all_a0[i], all_a1[i], all_a2[i], all_a3[i], all_a4[i], all_a5[i]]))

                        '''import matplotlib.pyplot as plt
                        full_length = len(self.timeseries_training_data)
                        plt.subplots(full_length, 1)
                        for month in range(full_length):
                            plt.subplot(full_length, 1, month+1)
                            print(self.timeseries_training_data[month][0]['month'])
                            plt.plot(self.timeseries_training_data[month][0]['observations'][4000:8000:10])'''
                        

                        np.save('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/home_power_holdout.npy', self.timeseries_holdout_data) 
                        np.save('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/home_power_training.npy', self.timeseries_training_data)

                        print("\n\nDONE loading and saving household data...")
                        import IPython
                        IPython.embed() 
                    else:

                        self.timeseries_holdout_data = np.load('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/home_power_holdout.npy')
                        self.timeseries_training_data = np.load('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/home_power_training.npy')             

                    '''import matplotlib.pyplot as plt
                    plot_this = self.timeseries_training_data[7][0]['actions']
                    plot_this_new = np.concatenate([np.expand_dims(plot_this[:,0], 1), plot_this[:,2:]], 1)
                    plt.subplot(4,1,1)
                    plt.plot(np.arange(plot_this.shape[0]), plot_this_new)
                    plt.subplot(4,1,2)
                    plt.plot(np.arange(plot_this.shape[0]/5), plot_this_new[0::5])
                    plt.subplot(4,1,3)
                    plt.plot(np.arange(plot_this.shape[0]/10), plot_this_new[0::10])
                    plt.subplot(4,1,4)
                    plt.plot(np.arange(plot_this.shape[0]/60), plot_this_new[0::60])
                    plt.show()'''

                    #for task_num in range(len(self.timeseries_training_data)):
                    #    print("Task num: ", task_num, " ... length: ", self.timeseries_training_data[task_num][0]['actions'].shape[0])
                    #i have about 900 "tasks" right now

                elif(self.which_timeseries=='stock'):

                    self.sampling_freq = 1 

                    #keys: ['symbol', 'open', 'close', 'low', 'high', 'volume']
                    #datapoints: 851264
                    if(need_to_save_data):

                        #get the companies
                        all_companies = np.unique(self.time_series_data.symbol) #501 of them
                        num_companies_train = int(len(all_companies)*0.9) #450
                        num_companies_val = int(len(all_companies)*0.1) #50 #not actually used...
                        companies_train = all_companies[:num_companies_train]
                        companies_val = all_companies[num_companies_train:]

                        #data move here
                        all_symbols = np.copy(self.time_series_data.symbol)
                        all_s0 = np.copy(self.time_series_data.open)
                        all_s1 = np.copy(self.time_series_data.close)
                        all_s2 = np.copy(self.time_series_data.low)
                        all_s3 = np.copy(self.time_series_data.high)

                        #first, make a dict of dicts
                            #key = name of company
                            #value = task_dict with observations, actions, agent_infos, rewards, env_infos
                        full_dict=dict()
                        print("Adding an entry for each company...")
                        for company in all_companies:
                            full_dict[company] = dict(observations=[], actions=[], agent_infos=0, rewards=0, env_infos=0, company=company)

                        #add all datapoints to the right entry of the dict, based on company
                        print("Reading in each data point...")
                        num_data = all_symbols.shape[0]
                        for i in range(num_data):
                            if(i%10000==0):
                                print("    ", i, "/", num_data)
                            curr_company = all_symbols[i]
                            full_dict[curr_company]['observations'].append([all_s0[i], all_s1[i], all_s2[i], all_s3[i]])
                            full_dict[curr_company]['company'] = curr_company

                        #go through full dict, and transfer its entry either to training or holdout data
                        self.timeseries_holdout_data = []
                        self.timeseries_training_data = []
                        print("Formatting each company's dict and putting it into train/val...")
                        for company in all_companies:
                            full_dict[company]['observations'] = np.array(full_dict[company]['observations'])[::self.sampling_freq] #check this dim
                            full_dict[company]['actions'] = np.array(full_dict[company]['observations'])[::self.sampling_freq]
                            if(company in companies_train):
                                self.timeseries_training_data.append([full_dict[company]])
                            else:
                                self.timeseries_holdout_data.append([full_dict[company]])

                        np.save('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/stock_holdout.npy', self.timeseries_holdout_data) 
                        np.save('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/stock_training.npy', self.timeseries_training_data)
                        
                        print("\n\nDONE loading and saving stock data...")
                        import IPython
                        IPython.embed()
                    else:
                        self.timeseries_holdout_data = np.load('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/stock_holdout.npy')
                        self.timeseries_training_data = np.load('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/stock_training.npy') 

                        #looks like dont need to sample it
                        '''import matplotlib.pyplot as plt
                        plot_this = self.timeseries_training_data[7][0]['actions']
                        plt.subplot(4,1,1)
                        plt.plot(np.arange(plot_this.shape[0]), plot_this)
                        plt.subplot(4,1,2)
                        plt.plot(np.arange(plot_this.shape[0]/5), plot_this[0::5])
                        plt.subplot(4,1,3)
                        plt.plot(np.arange(plot_this.shape[0]/10), plot_this[0::10])
                        plt.subplot(4,1,4)
                        plt.plot(np.arange(plot_this.shape[0]/60), plot_this[0::60])
                        plt.show()'''

                        '''import matplotlib.pyplot as plt
                        full_length = 18 ####len(self.timeseries_training_data)
                        plt.subplots(full_length, 1)
                        for task in range(full_length):
                            plt.subplot(full_length, 1, task+1)
                            print(self.timeseries_training_data[task][0]['company'])
                            plt.plot(self.timeseries_training_data[task][0]['observations'][-1000:])'''

                elif(self.which_timeseries=='sin'):

                    self.sampling_freq = 1

                    if(need_to_save_data):

                        ## data: ['t', 'value', 'taskID']
                        ## output: index [1] (of next timestep) - dim 1
                        ## inputs: index [1] (of curr timestep + of prev timestep) - dim 2 (make this s =1, a=1)

                        all_values = np.copy(self.time_series_data.value) #shape(numdata,)
                        all_taskIDs = np.copy(self.time_series_data.taskID) #shape(numdata,)

                        #make each "task" be a different f/A setting
                        self.timeseries_holdout_data = []
                        self.timeseries_training_data = []
                        
                        prev_task = -7
                        s_thistask = []
                        a_thistask = []

                        for i in range(all_values.shape[0]):

                            curr_task = all_taskIDs[i]

                            #STARTING A NEW TASK
                            if(not(curr_task==prev_task)): ###### TO DO: the prev task vs curr task might be a typo above too

                                print("looking at new taskID: ", curr_task, " ..... data point ", i, "/", all_values.shape[0])

                                if(i>0):
                                    #done collecting all data for prev task, so turn each thing into array and make a dict
                                    this_task_dict = dict(observations=np.array(s_thistask)[::self.sampling_freq], actions=np.array(a_thistask)[::self.sampling_freq], 
                                                        agent_infos=0, rewards=0, env_infos=0, taskID=prev_task)
                                    
                                    #add this to either training or validation, depending on taskID
                                    if(prev_task%10==0): ################# TO DO: pick this based on total #tasks
                                        self.timeseries_holdout_data.append([this_task_dict])
                                    else:
                                        self.timeseries_training_data.append([this_task_dict])
                                    
                                    #start a new task list
                                    s_thistask = []
                                    a_thistask = []

                                    ##actually, dont add new data until next timestep of this new task (so can have previous)
                                    ##s_thistask.append(all_values[i])
                                    ##a_thistask.append(all_values[i-1])

                                #update what task you're on
                                prev_task = curr_task

                            else: 
                                s_thistask.append(all_values[i])
                                a_thistask.append(np.array([all_values[i-1]]))

                        '''import matplotlib.pyplot as plt
                        full_length = len(self.timeseries_training_data)
                        plt.subplots(full_length, 1)
                        for taskID in range(full_length):
                            plt.subplot(full_length, 1, taskID+1)
                            print(self.timeseries_training_data[taskID][0]['taskID'])
                            plt.plot(self.timeseries_training_data[taskID][0]['observations'])
                        plt.show()'''

                        np.save('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/sin_holdout.npy', self.timeseries_holdout_data) 
                        np.save('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/sin_training.npy', self.timeseries_training_data)

                        print("\n\nDONE loading and saving sin data...")
                        import IPython
                        IPython.embed() 
                    else:

                        self.timeseries_holdout_data = np.load('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/sin_holdout.npy')
                        self.timeseries_training_data = np.load('/home/nagaban2/rllab_continuous/sandbox/ignasi/maml/data/sin_training.npy')   

                else:
                    print("\n\nERROR- this timeseries task not implemented in sampler")
                    import IPython
                    IPython.embed()

                    
                    

            #select mbs RANDOM training tasks to be returned
            #for each of those months, returns 1 rollout
            #each rollout is chunk of size 2000
            data_to_return=[]
            list_selected_tasks = np.random.randint(0,  len(self.timeseries_training_data), self.meta_bs)
            for i in list_selected_tasks:
                data_to_return.append(self.timeseries_training_data[i])
            #data_to_return[metabs][0]['actions'] : (44639, 6)
            #data_to_return[metabs][0]['observations'] : (44639, )
            data_to_return=np.array(data_to_return)

        else:       
            data_to_return = self.sampler.obtain_samples(max_path_length=self.max_path_length, batch_size=use_batchsize, num_updates=num_updates)

        return data_to_return

    #process the data in "paths"
        #this returns list of length = meta_bs (# envs/tasks)
        #each of those has 9 entries (observations, etc.)
                #each of those entries is of length = all datapoints collected for that task
    def process_samples(self, itr, paths):

        '''if(not_doing_mbrl):
            new_paths=[]
            for index in range(len(paths)):
                new_paths.append(paths[index][0]) #### only had 1 rollout, so appending them is just picking out that one
            return new_paths
        else:'''
        return self.sample_processor.process_samples(itr, paths, self.not_doing_mbrl, self.which_timeseries)

    def generate(self, firstTime=False, num_updates=1):

        ##############################################
        ## NOTE: everywhere it says "update_bs" in this func,
        ## it actually means update_bs*2 for maml, because that's how we call it
        ## update_bs will be used for training data for inner-loop gradient
        ## update_bs will be used for testing data for outer-loop gradient
        ##############################################
        new_data = False

        #set the policy to perform "random" actions
        if self.n_itr_rand > 0:
            try:
                self.policy.random = True
            except:
                raise (NotImplementedError, "The policy can't perform random actions")

        ###########
        ### TEMP

        optimize_policy=False
        self.policy.nn_policy=False
        if(self.itr>self.n_itr_rand):
            self.policy.random=False
        else:
            self.policy.random=True
        ###########

        '''if self.train_policy and self.itr % self.n_itr_policy == 0:
            self.policy.nn_policy = False
            optimize_policy = True
        elif self.train_policy:
            self.policy.nn_policy = True
            optimize_policy = False
        else:
            optimize_policy = False'''

        ###############################################
        ## generate samples
        ## if dataset is empty, 
        ## or if you already trained enough times on the current dataset
        ###############################################
        if ((self._data is None) or (self._epochs >= self.max_epochs)):

            print("\n\ngetting new training data to sample from, because sampled enough from current data")

            new_data = True
            #until you hit this many iters, perform rollouts using random actions
            #after that, perform rollouts using MPC controller
            if self.n_itr_rand > 0 and self.itr >= self.n_itr_rand:
                self.policy.random = False

                #perform rollouts
                #_data is of length = # of tasks sampled
                #each entry is a processed dict with 9 entries (obs, acs, returns, etc.)
                #each dict entry contains all of the sampled datapoints for the given task
            #print("\n\nSAMPLING DATA.... random: ", self.policy.random, "\n\n")
            _data = self.sample(self.itr, firstTime, num_updates)

            #append data from new rollouts into self._data 
            if self._data is None:
                self._data = _data
            else:
                self._data = self._data[-(self.max_tasks-self.meta_bs):] #this only matters if max_tasks < meta_bs
                self._data.extend(_data)

                #after 1st iteration (when you collected lot of data, to warmstart the regressor training),
                #count the total number of data points PER TASK as though you didnt collect extra during the 1st round

            #update total number of data points collected PER TASK (across all meta iterations of performing rollouts)
            dict_data = self.process_samples(self.itr, _data)
                #dict_data is a dict, where each entry has points from all rollouts of that task (for notmbrl, same as _data but formatted diff)

            nb = len(dict_data['obs_act'])
            x_mean = np.mean(dict_data['obs_act'], axis=0).reshape((1, -1))
            x_std = np.std(dict_data['obs_act'], axis=0).reshape((1, -1)) + 1e-8
            y_mean = np.mean(dict_data['diff_observations'], axis=0).reshape((1, -1))
            y_std = np.std(dict_data['diff_observations'], axis=0).reshape((1, -1)) + 1e-8
            self.regressor.update_params_data_dist(x_mean=x_mean, x_std=x_std,
                                                   y_mean=y_mean, y_std=y_std, nb=nb)
            #update mean/std stats using the mean/std of the newly collected data
            print("\n\nFINISH DATA COLLECTION FOR AGG ITER ", self.itr)
            print("PREV datapoints per task: ", self.n_data_points)
            self.n_data_points = np.minimum(int(1e7/self.meta_bs), self.n_data_points + len(dict_data['observations'])/self.meta_bs)
            print("CURR datapoints per task: ", self.n_data_points, "\n\n")
            
            #increment "itr" each time we do this data collection
            logger.dump_tabular(with_prefix=False)
            self.itr += 1   
            self._counter = 0
            self._epochs = 0

            if not self.correlated_inputs:
                if not self._dict_data:
                    self._dict_data = dict_data
                else:
                    self._dict_data = tensor_utils.concat_tensor_dict_list([dict_data, self._dict_data])

        ###########################################################
        ## define one epoch
        ## as sampling "update_bs" points "n_data_points/update_bs" times
        ###########################################################

        #if (self._counter + 1) * self.update_bs > self.n_data_points:
        if ((self._counter + 1) * self.update_bs) > (self.n_data_points/2.0):
            self._epochs += 1
            self._counter = 0

        ###############################################
        ## sample "update_bs" points from "meta_bs" data
        ###############################################

        #random list containing self.meta_bs items, each entry telling you "which task to look at"
        idxs_task=[]
        #random list containing self.meta_bs items, each entry telling you "which datapoints" (within that task)
        idxs_ts=[]
        idxs_path = []

        if ((not self.rnn) and self.correlated_inputs):
            for task_num in range(self.meta_bs):
                #which task
                which_task = np.random.randint(0,  len(self._data))
                idxs_task.append(which_task)
                which_path = np.random.randint(0, len(self._data[which_task]['observations']))
                idxs_path.append(which_path)

                chunk=self._data[which_task]['observations'][which_path]
                while(len(chunk)<=(self.update_bs + self.multi_input)):
                    which_task+=1
                    if(which_task>=len(self._data)):
                        which_task=0

                    which_path = np.random.randint(0, len(self._data[which_task]['observations']))
                    chunk = self._data[which_task]['observations'][which_path]

                    idxs_task[-1]=which_task
                    idxs_path[-1]=which_path

                #which datapoints
                i = np.random.randint(self.update_bs + self.multi_input, len(chunk))
                idxs_ts.append(np.arange(i - self.update_bs, i))

            #sample update_bs points from meta_bs tasks
            if self.oneshot:
                y = np.array([self._data[task]['diff_observations'][path][ts[-1]] for task, path, ts in
                              zip(idxs_task, idxs_path, idxs_ts)])
                x = np.array([self._data[task]['obs_act'][path][ts] for task, path, ts in
                              zip(idxs_task, idxs_path, idxs_ts)]).transpose((1, 0, 2))
            else:
                y = np.array([self._data[task]['diff_observations'][path][ts] for task, path, ts in zip(idxs_task, idxs_path, idxs_ts)])
                if self.multi_input:
                    x = np.array([[self._data[task]['obs_act'][path][t-self.multi_input+1:t+1].reshape(-1) for t in ts] for task, path, ts in zip(idxs_task, idxs_path, idxs_ts)])
                else:
                    x = np.array([self._data[task]['obs_act'][path][ts] for task, path, ts in zip(idxs_task, idxs_path, idxs_ts)])

        elif ((not self.rnn) and (not self.correlated_inputs)):
            idxs_ts = np.random.randint(0, len(self._dict_data['obs_act']), self.update_bs)
            y = [self._dict_data['diff_observations'][idxs_ts]]
            x = [self._dict_data['obs_act'][idxs_ts]]

        else:
            idxs_task = np.random.randint(0, len(self._data), self.meta_bs)
            idxs_path = np.array([np.random.randint(0, len(self._data[task]['obs_act'])) for task in idxs_task])

            useThis_max_path_length = self.max_path_length ############# 200 ###########self.max_path_length/self.sampling_freq

            if(self.not_doing_mbrl):
                x=[]
                y=[]
                for task in idxs_task:
                    chunk = self._data[task]['diff_observations'][0][:useThis_max_path_length]
                    while(len(chunk)<useThis_max_path_length):
                        task+=1
                        if(task>(len(self._data)-1)):
                            task=0 
                        chunk = self._data[task]['diff_observations'][0][:useThis_max_path_length] #pass basically the whole chunk to rnn (unlike gbac that takes 2k)
                    y.append(chunk)
                    x.append(self._data[task]['obs_act'][0][:useThis_max_path_length])

                y = np.array(y)
                x = np.array(x)
            else:
                y = np.array([self._data[task]['diff_observations'][path][:] for task, path in zip(idxs_task, idxs_path)])
                x = np.array([self._data[task]['obs_act'][path][:] for task, path in zip(idxs_task, idxs_path)])

            #increment counter each time you sample "update_bs" points from the dataset
            #this counter is reset when dataset is updated (when you collect more data)
        self._counter += 1

        ###############################################
        ## return "meta_bs" samples, each of "update_bs" points
        ## sampled from [s,a] and [s_diff]
        ## ie the inputs and outputs of dynamics model
        ###############################################

        return x, y

    #perform rollouts to collect samples
    def sample(self, itr, firstTime=False, num_updates=1):
        self.start_worker()
        start_time = time.time()
        itr_start_time = time.time()

        if(self.not_doing_mbrl==False):
            #sample a task for each of the envs
            _ = [env._wrapped_env._wrapped_env.reset_task() for env in self.sampler.vec_env.envs]

        with logger.prefix('itr #%d | ' % itr):

            #perform rollouts
            logger.log("Obtaining samples...")
            paths = self.obtain_samples(itr, firstTime, num_updates)

            ####### paths is:
                #list of length = mbs
                #each entry = list of rollouts
                #each rollout = dict of ['observations', 'actions', 'agent_infos', 'rewards', 'env_infos']

            #process them and return dicts
            logger.log("Processing samples...")
            samples_data = self.process_samples(itr, paths)

            ####### samples_data is:
                #list of length = mbs
                #each entry = 
                    #dict of ['obs_act', 'observations', 'undiscounted_returns', 'diff_observations', 'actions', 'rewards', 'returns']
                    #containing points from all rollouts

            logger.record_tabular('Time', time.time() - start_time)
            logger.record_tabular('ItrTime', time.time() - itr_start_time)
            # if self.plot:
            #     self.update_plot()
            #     if self.pause_for_plot:
            #         input("Plotting evaluation run: Press Enter to "
            #               "continue...")

        self.shutdown_worker()
        return samples_data

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        return dict(
            itr=itr,
            regressor=self.regressor,
            env=self.env,
        )

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    # def update_plot(self):
    #     if self.plot:
    #         plotter.update_plot(self.policy, self.max_path_length)
