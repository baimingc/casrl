
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import copy


##############################################
##############################################

def softmax_loss(loss, temperature):
    return np.exp(-loss/temperature)

def prior_weighting(t, this_task, past_probabilities, alpha, isnew=False):

    prior = 'CRP' #CRP, uniform

    if(prior=='CRP'):

        refreshing_freq = 20 ###20 for sin ###########200 ## this 100 and temp 0.02 and alpha 5
        new_effective_t = t-refreshing_freq*(int(t/refreshing_freq))
        starting_t = refreshing_freq*(int(t/refreshing_freq))

        if(isnew or (new_effective_t==0)):
            num = alpha
        else:
            num = 0
            for timestep in range(starting_t, t):
                #if this task existed at that timestep (else, probability is 0)
                if(this_task<len(past_probabilities[timestep])):
                    num += past_probabilities[timestep][this_task]
        denom = (new_effective_t+1) - 1 + alpha
        ret_val = num/denom
        #print(ret_val, "\n")
        return ret_val

    if(prior=='uniform'):
        return 1.0

##############################################
##############################################

def EM_iteration(obs, acts, next_obs, path_length, model, sess, 
                            thetaStar, tempLR, tally, temperature, alpha, k, 
                            past_probabilities, theta_old, continual,
                            givingDicts=False, firstHalf_feed_dict_inp=None, 
                            secondHalf_feed_dict_inp=None, max_num_thetas=100000,
                            should_pause=False, update_lr=0.01):

    em_frequency = 1## 4 #######8

    do_live_plot = False ##False
    timestep = path_length - (2*k+1)

    if(timestep%em_frequency==0):

        if(continual):
            input_tensors = [model.preloss, model.curr_gradients]
            input_tensor = [model.preloss]
        else:
            input_tensors = [model.task_lossa_useLater, model.curr_gradients]
            input_tensor = [model.task_lossa_useLater]

        if(givingDicts):
            firstHalf_feed_dict = firstHalf_feed_dict_inp
            secondHalf_feed_dict = secondHalf_feed_dict_inp

        else:
            #turn data into inputs and labels
            labels = (next_obs - obs).reshape(1, 2*k, -1)
            inputs = np.concatenate([obs, acts], axis=-1).reshape(1, 2*k, -1)
            
            if(continual):

                #split past K into 2 chunks
                firstHalf_inputs = inputs[:,0:k,:]
                firstHalf_labels = labels[:,0:k,:]
                secondHalf_inputs = inputs[:,k:,:]
                secondHalf_labels = labels[:,k:,:]

                firstHalf_feed_dict = {model.inputa_kpts: firstHalf_inputs, model.labela_kpts: firstHalf_labels}
                secondHalf_feed_dict = {model.inputa_kpts: secondHalf_inputs, model.labela_kpts: secondHalf_labels}
            else:

                #split past K into 2 chunks (but copy them twice so batchsize is still the same)
                firstHalf_inputs = np.concatenate([inputs[:,0:k,:],inputs[:,0:k,:]],axis=1)
                firstHalf_labels = np.concatenate([labels[:,0:k,:],labels[:,0:k,:]],axis=1)
                secondHalf_inputs = np.concatenate([inputs[:,k:,:],inputs[:,k:,:]],axis=1)
                secondHalf_labels = np.concatenate([labels[:,k:,:],labels[:,k:,:]],axis=1)

                firstHalf_feed_dict = {model.inputa: firstHalf_inputs, model.labela: firstHalf_labels}
                secondHalf_feed_dict = {model.inputa: secondHalf_inputs, model.labela: secondHalf_labels}

        list_of_losses=[]
        list_of_gradients=[]

        #get exp(-L_theta_T(t)/temp) using the past k points
        for curr_theta in theta_old:
            model.regressor.set_params(curr_theta)
            curr_loss, curr_grad = sess.run(input_tensors, feed_dict=secondHalf_feed_dict)
            list_of_losses.append(-curr_loss/temperature)
            list_of_gradients.append(curr_grad)

        #E step: use likelihood and CRP prior to calculate posterior task probability
        task_probabilities = []
        max_numerator = np.max(list_of_losses)
        for index in range(len(theta_old)):
            this_task_prob = np.exp(list_of_losses[index] - max_numerator)*prior_weighting(timestep, index, past_probabilities, alpha)
            task_probabilities.append(this_task_prob)
        task_probabilities = np.array(task_probabilities)/np.sum(task_probabilities)

        #M step: use task probabilities to update task paramaters for each task
        theta_new=[]
        for index in range(len(theta_old)): 
            old_theta = copy.deepcopy(theta_old[index])
            ###for _ in range(em_frequency):
            old_theta = dict(zip(old_theta.keys(), [old_theta[key] - task_probabilities[index] * tempLR[index] * list_of_gradients[index][key] for key in old_theta.keys()]))
            theta_new.append(old_theta)

        #adapt theta* by taking a gradient step using the 1st k points
        model.regressor.set_params(thetaStar)
        curr_loss, curr_grad = sess.run(input_tensors, feed_dict=firstHalf_feed_dict)
        this_task_prob = 1.0
        theta_additional = dict(zip(thetaStar.keys(), [thetaStar[key] - this_task_prob * update_lr * curr_grad[key] for key in thetaStar.keys()]))
        model.regressor.set_params(theta_additional)

        #eval loss of theta_additional on the 2nd k points
        curr_loss_additional, curr_grad_additional = sess.run(input_tensors, feed_dict=secondHalf_feed_dict)
        curr_loss_additional = -curr_loss_additional/temperature

        #### NEW: idk
        #E: recalculate all task probabilities, now that we added new task
        list_of_losses_temp = np.concatenate([list_of_losses, [curr_loss_additional]], 0)
        new_max_numerator = np.max(list_of_losses_temp)
        task_probabilities_temp=[]
        for index in range(len(list_of_losses_temp)):
            if(index==(len(list_of_losses_temp)-1)):
                this_task_prob = np.exp(list_of_losses_temp[index]- new_max_numerator)*prior_weighting(timestep, 0, past_probabilities, alpha, isnew=True)
            else:    
                this_task_prob = np.exp(list_of_losses_temp[index]- new_max_numerator)*prior_weighting(timestep, index, past_probabilities, alpha)
            task_probabilities_temp.append(this_task_prob)
        task_probabilities_temp = np.array(task_probabilities_temp)/np.sum(task_probabilities_temp)
        task_prob_new = task_probabilities_temp[-1]

        #if the loss of this theta_additional is better than all the other ones, add it on
        if(len(theta_old) < max_num_thetas):

            ########if(curr_loss_additional>np.max(list_of_losses)):
            ########if(task_prob_new>np.max(task_probabilities_temp[:-1])):
            if(np.argmax(task_probabilities_temp)==(len(task_probabilities_temp)-1)):

                #add as option to the list
                tempLR.append(update_lr)
                tally.append(0)
                theta_new.append(theta_additional)
                task_probabilities = task_probabilities_temp
                list_of_losses.append(curr_loss_additional)

                for index in range(len(theta_old)):

                    this_task_prob = task_probabilities[index]

                    #the new theta_T dist should use the new P, not the old one (replace the most recent update)
                    this_grad = list_of_gradients[index]
                    old_theta = copy.deepcopy(theta_old[index])
                    ###for _ in range(em_frequency):
                    old_theta = dict(zip(old_theta.keys(), [old_theta[key] - this_task_prob * tempLR[index] * this_grad[key] for key in old_theta.keys()]))
                    theta_new[index]=old_theta

                #adapt the new one too
                theta_new[-1] = dict(zip(theta_additional.keys(), [theta_additional[key] - task_probabilities[-1] * tempLR[-1] * curr_grad_additional[key] for key in theta_additional.keys()])) 

        if(do_live_plot):
            plt.clf()
            plt.ylim(0,1)
            plt.bar(np.arange(len(task_probabilities)), task_probabilities)
            plt.pause(0.005)

        #reset your regressor to the one from the list w the best loss, so you can use that for control
        best_theta_option= theta_new[np.argmax(list_of_losses)]
        model.regressor.set_params(best_theta_option)

        #set new thetas to be old thetas, for next iter
        theta_old = copy.deepcopy(theta_new)

        #save the probabilities for this timestep, of each task
        past_probabilities.append(task_probabilities.tolist())
        #print("list of losses: ", list_of_losses)
        #print("task probabilities: ", task_probabilities)

        '''gamma= 0.5
        for i in range(len(tempLR)):
            tally[i] += task_probabilities[i]*tempLR[i]
            decay_criteria= tally[i]/(update_lr*5.0) ###every ~100 steps of size update_lr, cut lr in half ######## 5 mid 100 top
            tempLR[i] = tempLR[i]*np.power(gamma, decay_criteria)

            if(tally[i]>(update_lr*50.0)):
                print("set ", i, " to stop adapting...")
                tempLR[i] = 0 #################### stop updating after a while'''

        #print("t: ", timestep, " ... tallies: ", tally, " ... task probs: ", task_probabilities)
        #print("t: ", timestep, " .... LRs: ", tempLR)


        #decay = 10.0
        #tempLR = tempLR/(1+(timestep*decay)) ############################## decay lr according to something

    else:
        past_probabilities.append(past_probabilities[-1])

    return past_probabilities, theta_old, tempLR, tally
