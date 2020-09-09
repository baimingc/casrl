""" Code for the MAML algorithm and network definitions. """
import numpy as np
import tensorflow as tf
import copy

from mole.utils import mse

class MAML_continual:
    def __init__(self, regressor, k, validation_set_size, ignore_absolute_xy, agent_type,
                dim_input_regressor=1, dim_input=1, dim_output=1, dim_bias=1, config={}):
        """ must call construct_model() after initializing MAML! """
        self.k = k
        self.validation_set_size = validation_set_size
        self.dim_input_regressor = dim_input_regressor #sometimes lower than dim_input, because sometimes we ignore x/y/etc
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = config['update_lr']
        self.meta_lr = tf.placeholder_with_default(config['meta_lr'], ())
        self.loss_func = mse
        self.regressor = regressor
        self.regressor.construct_fc_weights(meta_loss=config['meta_loss'])
        self.forward = self.regressor.forward_fc
        self.config = config
        self.meta_learn_lr = False ####
        self.agent_type = agent_type
        self.ignore_absolute_xy = ignore_absolute_xy
        self.dim_bias = dim_bias

        self.max_num_steps = self.config['update_batch_size']-2*self.k

    def construct_model(self, input_tensors=None, prefix='metatrain_'):

        #placeholders to hold the inputs/outputs
        if input_tensors is None:

            #initialize placeholders needed only for the metatrain_op
            self.probabilities = tf.placeholder(tf.float32, shape=(self.config['update_batch_size']-2*self.k, ))
            self.inputa_rollout = tf.placeholder(tf.float32, shape=(self.config['update_batch_size'], self.dim_input))
            self.labela_rollout = tf.placeholder(tf.float32, shape=(self.config['update_batch_size'], self.regressor.dim_output))

            #initialize placeholders needed only for the test_op
            self.inputa_kpts = tf.placeholder(tf.float32, shape=(None, self.k, self.dim_input))
            self.labela_kpts = tf.placeholder(tf.float32, shape=(None, self.k, self.regressor.dim_output))
            
            #initialize placeholders for initial theta
            self.orig_theta_b0 = tf.placeholder(tf.float32, shape=(512,))
            self.orig_theta_b1 = tf.placeholder(tf.float32, shape=(512,))
            self.orig_theta_b2 = tf.placeholder(tf.float32, shape=(512,))
            self.orig_theta_b3 = tf.placeholder(tf.float32, shape=(self.regressor.dim_output,))
            if(self.ignore_absolute_xy):
                self.orig_theta_W0 = tf.placeholder(tf.float32, shape=(self.dim_input_regressor+self.dim_bias, 512)) 
            else:
                self.orig_theta_W0 = tf.placeholder(tf.float32, shape=(self.dim_input+self.dim_bias, 512)) 
            self.orig_theta_W1 = tf.placeholder(tf.float32, shape=(512,512))
            self.orig_theta_W2 = tf.placeholder(tf.float32, shape=(512,512))
            self.orig_theta_W3 = tf.placeholder(tf.float32, shape=(512,self.regressor.dim_output))
            self.orig_theta_bias = tf.placeholder(tf.float32, shape=(self.dim_bias,))

            #initialize placeholders for metatrain gradient step
            self.given_grad_b0 = tf.placeholder(tf.float32, shape=(512,))
            self.given_grad_b1 = tf.placeholder(tf.float32, shape=(512,))
            self.given_grad_b2 = tf.placeholder(tf.float32, shape=(512,))
            self.given_grad_b3 = tf.placeholder(tf.float32, shape=(self.regressor.dim_output,))
            if(self.ignore_absolute_xy):
                self.given_grad_W0 = tf.placeholder(tf.float32, shape=(self.dim_input_regressor+self.dim_bias, 512)) 
            else:
                self.given_grad_W0 = tf.placeholder(tf.float32, shape=(self.dim_input+self.dim_bias, 512)) 
            self.given_grad_W1 = tf.placeholder(tf.float32, shape=(512,512))
            self.given_grad_W2 = tf.placeholder(tf.float32, shape=(512,512))
            self.given_grad_W3 = tf.placeholder(tf.float32, shape=(512,self.regressor.dim_output))
            self.given_grad_bias = tf.placeholder(tf.float32, shape=(self.dim_bias,))

        else:
            #self.max_num_thetas = input_tensors['max_num_thetas']
            print("\n\nTO DO: case of having input_tensors... haven't finished typing this part in maml_continual...")
            import IPython
            IPython.embed()

        #placeholders to hold the preprocessed inputs/outputs (mean 0, std 1)
        inputa_kpts = (self.inputa_kpts - self.regressor._x_mean_var)/self.regressor._x_std_var
        labela_kpts = (self.labela_kpts - self.regressor._y_mean_var)/self.regressor._y_std_var
        inputa_rollout = (self.inputa_rollout - self.regressor._x_mean_var)/self.regressor._x_std_var
        labela_rollout = (self.labela_rollout - self.regressor._y_mean_var)/self.regressor._y_std_var

        with tf.variable_scope('model', reuse=None) as training_scope:
            #set thetaStar
            self.weights = weights = self.regressor.weights

            #init loop vars
            self.loss_for_this_theta = tf.constant(0.0)
            b0 = self.orig_theta_b0
            b1 = self.orig_theta_b1
            b2 = self.orig_theta_b2
            b3 = self.orig_theta_b3
            W0 = self.orig_theta_W0
            W1 = self.orig_theta_W1
            W2 = self.orig_theta_W2
            W3 = self.orig_theta_W3
            bias = self.orig_theta_bias
            
            #add the first 2k datapoints onto our list of seen points
            self.inputs = inputa_rollout[0:2*self.k]
            self.labels = labela_rollout[0:2*self.k]

            #########################################################
            ## take 1 gradient step away from this theta, at each timestep
            #########################################################

            for timestep in range(2*self.k, 2*self.k+self.max_num_steps):

                #get the probability of this task at this timestep
                this_prob = self.probabilities[timestep-2*self.k]

                #2 chunks, each w k data points
                inputs_first_k = self.inputs[-2*self.k:-self.k]
                labels_first_k = self.labels[-2*self.k:-self.k]
                inputs_second_k = self.inputs[-self.k:]
                labels_second_k = self.labels[-self.k:]

                #combine into a single dict
                this_theta= {'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3, 'W0': W0, 'W1': W1, 'W2': W2, 'W3': W3, 'bias': bias}

                #calculate gradient of loss of using this theta
                this_output = self.forward(inputs_second_k, this_theta, reuse=True, meta_loss=self.config['meta_loss'])
                this_loss = self.loss_func(this_output, labels_second_k)
                grads = tf.gradients(this_loss, list(this_theta.values()))
                this_grad = dict(zip(this_theta.keys(), grads))

                #adapt the theta
                b0 = this_theta['b0'] - this_prob*self.config['update_lr'] * this_grad['b0']
                b1 = this_theta['b1'] - this_prob*self.config['update_lr'] * this_grad['b1']
                b2 = this_theta['b2'] - this_prob*self.config['update_lr'] * this_grad['b2']
                b3 = this_theta['b3'] - this_prob*self.config['update_lr'] * this_grad['b3']
                W0 = this_theta['W0'] - this_prob*self.config['update_lr'] * this_grad['W0']
                W1 = this_theta['W1'] - this_prob*self.config['update_lr'] * this_grad['W1']
                W2 = this_theta['W2'] - this_prob*self.config['update_lr'] * this_grad['W2']
                W3 = this_theta['W3'] - this_prob*self.config['update_lr'] * this_grad['W3']
                bias = this_theta['bias'] - this_prob*self.config['update_lr'] * this_grad['bias']

                '''##########TEMP:
                adapted_theta= {'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3, 'W0': W0, 'W1': W1, 'W2': W2, 'W3': W3, 'bias': bias}
                val_output = self.forward(inputs_second_k, adapted_theta, reuse=True, meta_loss=self.config['meta_loss'])
                val_loss = self.loss_func(val_output, labels_second_k)'''

                #update return vars
                self.loss_for_this_theta += this_prob*this_loss

                #add to list of what you've seen
                self.inputs = tf.concat([self.inputs, tf.expand_dims(inputa_rollout[timestep],0)], axis=0)
                self.labels = tf.concat([self.labels, tf.expand_dims(labela_rollout[timestep],0)], axis=0)

            self.loss_for_this_theta /= self.max_num_steps

        #########################################
        ## Define gradients
        #########################################

        self.this_grad_b0 = tf.gradients(self.loss_for_this_theta, self.orig_theta_b0)[0] 
        self.this_grad_b1 = tf.gradients(self.loss_for_this_theta, self.orig_theta_b1)[0]
        self.this_grad_b2 = tf.gradients(self.loss_for_this_theta, self.orig_theta_b2)[0]
        self.this_grad_b3 = tf.gradients(self.loss_for_this_theta, self.orig_theta_b3)[0]
        self.this_grad_W0 = tf.gradients(self.loss_for_this_theta, self.orig_theta_W0)[0]
        self.this_grad_W1 = tf.gradients(self.loss_for_this_theta, self.orig_theta_W1)[0]
        self.this_grad_W2 = tf.gradients(self.loss_for_this_theta, self.orig_theta_W2)[0]
        self.this_grad_W3 = tf.gradients(self.loss_for_this_theta, self.orig_theta_W3)[0]
        self.this_grad_bias = tf.gradients(self.loss_for_this_theta, self.orig_theta_bias)[0]
                    
        #########################################
        ## Define test_op
        #########################################

        #Take 1 step away from current weights using loss(f_weights(inputa_kpts))

        #get data and weights
        self.curr_weights = weights

        #f_weights(inputa_kpts)
        task_outputa = self.forward(inputa_kpts[0][-self.k:], self.curr_weights, reuse=True, meta_loss=self.config['meta_loss'])

        self.checkthisoutp = task_outputa
        self.checkthislabel = labela_kpts[0][-self.k:]
        
        #loss(f_weights(inputa_kpts))
        self.preloss = self.loss_func(task_outputa, labela_kpts[0][-self.k:])
        
        # update weights = use loss(f_weights{0}(inputa_kpts)) to take a gradient step
        grads = tf.gradients(self.preloss, list(self.curr_weights.values()), name='grad_prelossWRTtheta')
        self.curr_gradients = dict(zip(self.curr_weights.keys(), grads))
        self.adapted_weights = dict(zip(self.curr_weights.keys(), [self.curr_weights[key] - self.update_lr * self.curr_gradients[key] for key in self.curr_weights.keys()]))
        self.test_op = [tf.assign(v, self.adapted_weights[k]) for k, v in weights.items()]

        #########################################
        ## Define metatrain_op
        #########################################

        list_grads = [self.given_grad_b0, self.given_grad_b1, self.given_grad_b2, self.given_grad_b3,
                    self.given_grad_W0, self.given_grad_W1, self.given_grad_W2, 
                    self.given_grad_W3, self.given_grad_bias]
        list_vars = [self.weights['b0'], self.weights['b1'], self.weights['b2'], self.weights['b3'], 
                    self.weights['W0'], self.weights['W1'], self.weights['W2'], 
                    self.weights['W3'], self.weights['bias']]

        self.optimizer = tf.train.AdamOptimizer(self.meta_lr)
        self.metatrain_op = self.optimizer.apply_gradients(zip(list_grads, list_vars))

        ##############################################
        ## Summaries
        ##############################################

        ###tf.summary.scalar(prefix+'loss for test_op', self.preloss)
        ###tf.summary.scalar(prefix+'loss for metatrain_op for this theta ', self.loss_for_this_theta)


