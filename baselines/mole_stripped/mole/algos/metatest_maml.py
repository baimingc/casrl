import numpy as np
from rllab.misc import tensor_utils
import time
import tensorflow as tf
import copy
import joblib
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from mole.em_iteration import EM_iteration

class MetaTest(object):

    def __init__(self, chunk_size_into_model, k, model, thetaResetChoice, tempLR, temp, dontResetBetwRollouts, 
                do_live_plot, filepath, testing_a_continuallyTrained_model, config, num_updates, tasktype=None):
        self.model = model
        self.update_lr = tempLR
        self.temp = temp
        self.dontResetBetwRollouts = dontResetBetwRollouts
        self.k = k

        #in this file, everything that says udpate_bs means (2k) for continual, and (k) for noncontinual
        self.update_bs = chunk_size_into_model 

        self.max_path_length = config['max_path_length'] + self.update_bs
        self.animated = config['animated']
        self.speedup = config['speedup']
        self.num_rollouts = config['num_rollouts']
        self.sess = tf.get_default_session()
        self.render_config = config['render']
        self.num_updates = num_updates
        self.update = False
        self.do_live_plot = do_live_plot
        self.filepath = filepath
        self.tasktype = tasktype
        self.testing_a_continuallyTrained_model = testing_a_continuallyTrained_model ### this means, what was it trained w?

        self.thetaResetChoice = thetaResetChoice
        #1: decide based on training loss on past K
        #2: always reset to theta*
        #3: always continue taking steps from curr theta
        #4: decide based on validation loss on past 0.5K
        #5: keep a task distribution for each timestep + a theta distribution over tasks
            #what task you're in is a soft decision now

    def update_regressor(self, obs, acts, next_obs, path_length, env, agent):

        #########################################

        if(self.thetaResetChoice==5): ##TEST w continual option

            alpha = 1 
            thetaStar = self._weights

            self.past_probabilities, self.theta_old, self.tempLR, self.tally = EM_iteration(obs, acts, next_obs, path_length, self.model, self.sess, thetaStar, 
                                                                            self.tempLR, self.tally, self.temp, alpha, self.k, self.past_probabilities, 
                                                                            self.theta_old, continual=self.testing_a_continuallyTrained_model,
                                                                            update_lr = self.update_lr, max_num_thetas=5)

            #if(path_length%10==0):
            #    print(self.past_probabilities[-1])

        #########################################

        else:

            #next_obs and obs are 2k... (cuz the testwcontinual option needs to receive 2k)
                #but if we are testing w/o continual, we only need the past k of these points to be passed into our model

            if(self.testing_a_continuallyTrained_model):

                labels = (next_obs - obs).reshape(1, self.update_bs, -1)
                inputs = np.concatenate([obs, acts], axis=-1).reshape(1, self.update_bs, -1)
                feed_dict = {self.model.inputa_kpts: inputs[:,-self.k:,:], self.model.labela_kpts: labels[:,-self.k:,:]}
                self.update = True

                input_tensors_testAndLoss= [self.model.test_op, self.model.preloss]
                input_tensor_loss= [self.model.preloss]

            else:
                labels = (next_obs - obs).reshape(1, self.update_bs, -1)
                inputs = np.concatenate([obs, acts], axis=-1).reshape(1, self.update_bs, -1)

                feed_dict = {self.model.inputa: inputs[:,-self.k:,:], self.model.labela: labels[:,-self.k:,:]}
                self.update = True

                input_tensors_testAndLoss= [self.model.test_op, self.model.task_lossa_useLater]
                input_tensor_loss= [self.model.task_lossa_useLater]

            ############################
            ######## ALWAYS RESET TO THETA*
            ############################

            if(self.thetaResetChoice==2):
                #restore theta*
                self.model.regressor.set_params(self._weights)

                #use past "update_bs" steps to take a gradient step (or multiple) away from theta*
                for _ in range(self.num_updates):
                    self.sess.run([self.model.test_op], feed_dict=feed_dict)

            ############################
            ######## ALWAYS CONTINUE ADAPTING
            ############################

            if(self.thetaResetChoice==3):
                #use past "update_bs" steps to take a gradient step (or multiple) away from curr theta
                for _ in range(self.num_updates):
                    self.sess.run([self.model.test_op], feed_dict=feed_dict)

            ############################
            ######## CHOOSE WHICH TO RESET TO (based on valid # on past 0.5K)
            ############################

            if(self.thetaResetChoice==4):

                #split past K into 2 chunks (but copy them twice so batchsize is still the same)
                firstHalf_inputs = np.concatenate([inputs[:,0:int(self.k/2),:],inputs[:,0:int(self.k/2),:]],axis=1)
                firstHalf_labels = np.concatenate([labels[:,0:int(self.k/2),:],labels[:,0:int(self.k/2),:]],axis=1)
                secondHalf_inputs = np.concatenate([inputs[:,int(self.k/2):,:],inputs[:,int(self.k/2):,:]],axis=1)
                secondHalf_labels = np.concatenate([labels[:,int(self.k/2):,:],labels[:,int(self.k/2):,:]],axis=1)

                if(self.testing_a_continuallyTrained_model):
                    firstHalf_feed_dict = {self.model.inputa_kpts: firstHalf_inputs, self.model.labela_kpts: firstHalf_labels}
                    secondHalf_feed_dict = {self.model.inputa_kpts: secondHalf_inputs, self.model.labela_kpts: secondHalf_labels}
                else:
                    firstHalf_feed_dict = {self.model.inputa: firstHalf_inputs, self.model.labela: firstHalf_labels}
                    secondHalf_feed_dict = {self.model.inputa: secondHalf_inputs, self.model.labela: secondHalf_labels}

                #the first option in self.list_of_theta options is already theta*
                #the second option in self.list_of_theta options should always be the curr_theta
                curr_theta = self.model.regressor.get_params()
                if(len(self.list_of_theta_options)==1):
                    self.list_of_theta_options.append(curr_theta)
                else:
                    self.list_of_theta_options[1] = curr_theta

                #decide whether to adapt from curr theta or from theta*
                    #option --> adapt it using 1st 0.5K pts --> select the adapted option that does best on 2nd 0.5K pts
                options=[]
                losses_of_options=[]
                for theta_option in self.list_of_theta_options:
                    #restore to option
                    self.model.regressor.set_params(theta_option)

                    #use 1st chunk of 0.5K steps to take a gradient step (or multiple) away from option
                    for _ in range(self.num_updates):
                        _, trainingloss = self.sess.run(input_tensors_testAndLoss, feed_dict=firstHalf_feed_dict)

                    #check validation loss of new model params on 2nd chunk of 0.5K steps
                    validationloss = self.sess.run(input_tensor_loss, feed_dict=secondHalf_feed_dict)[0]

                    #save these adapted params, so can restore to it later if it's the best
                    options.append(self.model.regressor.get_params())

                    #save the loss of this new adapted option (on those K pts)
                    losses_of_options.append(np.mean(validationloss))

                #change your regressor to that best adapted one
                best_index = np.argmin(np.array(losses_of_options))
                option = options[best_index]
                self.model.regressor.set_params(option)

                #if theta* option was the best, mark that as a task change... save curr theta into list for future
                if(best_index==0):
                    junk=1
                    self.times_thetaStar+=1
                    #print("...RESET TO THETA* ... step ", path_length)
                    #########################################self.list_of_theta_options.append(curr_theta)
                elif(best_index==1):
                    junk=1
                    self.times_continue+=1
                    #print("...CONTINUE TAKING STEPS FROM CURR THETA")
                else:
                    junk=1
                    #print("...RESET TO SAVED OPTION #", (best_index-1))


        ########################################################
        ########################################################
        ########################################################

    def metatest(self, env, agent):
        returns  = []
        distances = []
        zs = []

        #save original weights = the optimal weights from training across many tasks = theta*
        self._weights = self.model.regressor.get_params()


        for i in range(self.num_rollouts):

            if((i==0) or (self.dontResetBetwRollouts==False)):
                #restart our list of options
                self.list_of_theta_options = []
                self.theta_old = []
                self.tempLR = []
                self.tally = []
                #start both of these lists with this theta* from metatraining
                self.list_of_theta_options.append(self._weights)
                self.theta_old.append(self._weights)
                self.tempLR.append(self.update_lr)
                self.tally.append(0)

                self.past_probabilities = []

                self.times_thetaStar=0
                self.times_continue=0

            #perform a rollout
            print('\n--------------------------------------\nRollout num  : ', i)

            path = self.rollout(env, agent)
            R = sum(path["rewards"][self.update_bs:])
            returns.append(R)
            print('Total reward  : ',  R)
            ####print('Theta reset choice  : ',  self.thetaResetChoice)
            ####print('# Times theta*  : ',  self.times_thetaStar)
            ####print('# Times theta  : ',  self.times_continue)

            #SAVE FOR REPLAY
            np.save(self.filepath + str(self.tasktype) + '_n' +str(self.thetaResetChoice) + '_rollout' + str(i) + '_startingstate.npy', path['starting_state'])
            np.save(self.filepath + str(self.tasktype) + '_n' +str(self.thetaResetChoice) + '_rollout' + str(i) + '_actions.npy', path['actions'])
            np.save(self.filepath + str(self.tasktype) + '_n' +str(self.thetaResetChoice) + '_rollout' + str(i) + '_probabilities.npy', self.past_probabilities)
            np.save(self.filepath + str(self.tasktype) + '_n' +str(self.thetaResetChoice) + '_rollout' + str(i) + '_ubs.npy', self.update_bs)
            np.save(self.filepath + str(self.tasktype) + '_n' +str(self.thetaResetChoice) + '_rollout' + str(i) + '_rewards.npy', path["rewards"])

            returns.append(R)
            distances.append(path['observations'][-1][-3])
            zs.append(path['observations'][-1][-1])

            #reset weights back to theta* at the end of the rollout
            if(self.dontResetBetwRollouts):
                pass
            else:
                self.model.regressor.set_params(self._weights)

        print('\n--------------------------------------')
        print('MEAN_returns:   ', np.mean(returns))
        print('STD_returns:    ', np.std(returns))
        print('MEAN_xs:        ', np.mean(distances))
        print('STD_xs:         ', np.std(distances))
        print('frac_z:         ', sum([int(z > 0.75) for z in zs]) / len(zs))
        print("ubs: ", self.update_bs)

    def rollout(self, env, agent):

        #time to sleep between each render
        timestep = env.wrapped_env.wrapped_env.model.opt.timestep
        timestep_updated = timestep / self.speedup


        #sample a task
        env._wrapped_env._wrapped_env.reset_task()

        #init vars
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        dones = []
        o = env.reset()
        starting_state = env._wrapped_env._wrapped_env.my_state

        observations.append(o)
        agent.reset()
        path_length = 0
        if(self.animated):
            env.render(config=self.render_config)
        t0 = time.time()

        #perform a rollout
        while path_length < self.max_path_length:

            #if(self.thetaResetChoice==0 or self.thetaResetChoice==5):
            if(path_length%200==0):
                print("   step: ", path_length)

            #policy gives action
            a, agent_info = agent.get_action(o)

            #execute action (one step)
            env.wrapped_env.wrapped_env.step_num=path_length
            next_o, r, d, env_info = env.step(a)

            #save info about that step
            observations.append(env.observation_space.flatten(o))
            rewards.append(r)
            actions.append(env.action_space.flatten(a))
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            dones.append(d)

            #update number of steps taken
            path_length += 1

            #check if done
            if d:
                break

            #use the last "update_bs" steps to update weights of dynamics model
            if path_length > self.update_bs:
                obs = np.array(observations)[-self.update_bs:]
                acts = np.array(actions)[-self.update_bs:]
                next_obs = np.array(observations + [next_o])[-self.update_bs:]
                self.update_regressor(obs, acts, next_obs, path_length, env, agent)

            #update observation
            o = next_o

            #render
            if(self.animated):
                if(path_length>0):
                    env.render(config=self.render_config)
                    time.sleep(timestep_updated)
        if(self.animated):
            env.render(close=True, config=self.render_config)

        # visualize live plot of task probabilities
        if(self.do_live_plot):
            plt.show()

        return dict(
            observations=tensor_utils.stack_tensor_list(observations),
            actions=tensor_utils.stack_tensor_list(actions),
            rewards=tensor_utils.stack_tensor_list(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            dones=np.asarray(dones),
            last_obs=o,
            starting_state=starting_state,
        )

    def take_k_steps(self, env, agent, numSteps):

        #time to sleep between each render
        timestep = env.wrapped_env.wrapped_env.model.opt.timestep
        timestep_updated = timestep / self.speedup

        #init vars
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        dones = []

        #get current obs
        o = env.wrapped_env.get_current_obs()
        observations.append(o)

        path_length = 0
        t0 = time.time()

        #perform a rollout
        while path_length < numSteps:

            #policy gives action
            a, agent_info = agent.get_action(o)

            #execute action (one step)
            next_o, r, d, env_info = env.step(a)

            #save info about that step
            observations.append(env.observation_space.flatten(o))
            rewards.append(r)
            actions.append(env.action_space.flatten(a))
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            dones.append(d)

            #update number of steps taken
            path_length += 1

            #check if done
            if d:
                break

            #update observation
            o = next_o

        return dict(
            observations=tensor_utils.stack_tensor_list(observations),
            actions=tensor_utils.stack_tensor_list(actions),
            rewards=tensor_utils.stack_tensor_list(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            dones=np.asarray(dones),
            last_obs=o,
        )
