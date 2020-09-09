""" Code for the MAML algorithm and network definitions. """
import numpy as np
import tensorflow as tf
import copy

from mole.utils import mse

class MAML:
    def __init__(self, regressor, dim_input=1, dim_output=1, num_extra=0, config={}):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = config['update_lr']
        self.meta_lr = tf.placeholder_with_default(config['meta_lr'], ())
        ############self.num_updates = config['num_updates']
        self.loss_func = mse
        self.regressor = regressor
        self.regressor.construct_fc_weights(meta_loss=config['meta_loss'])
        self.forward = self.regressor.forward_fc
        self.config = config
        self.meta_learn_lr = config.get('meta_learn_lr', False)

        self.num_extra = num_extra


    def construct_model(self, input_tensors=None, prefix='metatrain_'):

        #placeholders to hold the inputs/outputs
            # a: training data for inner gradient (this will be split into multiple chunks, so can do SGD)
            # b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32, shape=(None, self.config['update_batch_size']+self.num_extra, self.dim_input))
            self.inputb = tf.placeholder(tf.float32, shape=(None, self.config['update_batch_size'], self.dim_input))
            self.labela = tf.placeholder(tf.float32, shape=(None, self.config['update_batch_size']+self.num_extra, self.regressor.dim_output))
            self.labelb = tf.placeholder(tf.float32, shape=(None, self.config['update_batch_size'], self.regressor.dim_output))
            self.num_sgd_steps = tf.placeholder(tf.int32)
            self.num_updates = tf.placeholder(tf.int32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']
            self.num_sgd_steps = input_tensors['num_sgd_steps']
            self.num_updates = input_tensors['num_updates']

        #placeholders to hold the preprocessed inputs/outputs (mean 0, std 1)
        inputa = (self.inputa - self.regressor._x_mean_var)/self.regressor._x_std_var
        inputb = (self.inputb - self.regressor._x_mean_var)/self.regressor._x_std_var
        labela = (self.labela - self.regressor._y_mean_var)/self.regressor._y_std_var
        labelb = (self.labelb - self.regressor._y_mean_var)/self.regressor._y_std_var
        
        with tf.variable_scope('model', reuse=None) as training_scope:

            #with sess.as_default():
            #    num_updates=self.num_updates.eval()

            #define the weights to be the regressor weights... these are weights{0}
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.regressor.weights

            if self.meta_learn_lr:
                self.update_lr = dict([(k, tf.Variable(self.update_lr * tf.ones(tf.shape(v)),
                                                       name='lr_'+k)) for k,v in weights.items()])

            #########################################################
            ## Function to perform the inner gradient calculations
            #########################################################

            def innerLoop(a, b, la, lb, j, curr_weights, task_lossesb):
                i=tf.constant(0)
                loop_result = tf.while_loop(condition, takeAStep, [a, b, la, lb, i, curr_weights, task_lossesb], shape_invariants=[a.get_shape(), b.get_shape(), la.get_shape(), lb.get_shape(), i.get_shape(), self.shapes, tf.TensorShape([None])])
                curr_weights=loop_result[5]
                # L(f_thetaLast(b))
                output = self.forward(b, curr_weights, reuse=True, meta_loss=False)

                #################task_outputbs.append(output)
                #################task_lossesb.append(self.loss_func(output, lb))
                task_lossesb = tf.concat([task_lossesb, [self.loss_func(output, lb)]], 0)

                #return and increment outercounter
                return [a, b, la, lb, j+1, curr_weights, task_lossesb]

            def takeAStep(a, b, la, lb, i, curr_weights, task_lossesb):
                k=self.config['update_batch_size']
                #f_theta_i(inputa_chunk_i)
                task_outputa = self.forward(a[-k-self.num_sgd_steps+i:-self.num_sgd_steps+i], curr_weights, reuse=True, meta_loss=self.config['meta_loss'])  # only reuse on the first iter
                #L(f_theta_i(inputa_chunk_i))
                task_lossa = self.loss_func(task_outputa, la[-k-self.num_sgd_steps+i:-self.num_sgd_steps+i])

                # theta_i+1 = use L(f_theta_i(inputa_chunk_i))) to take a gradient step on theta_i
                grads = tf.gradients(task_lossa, list(curr_weights.values()))
                gradients = dict(zip(curr_weights.keys(), grads))
                curr_weights = dict(
                    zip(curr_weights.keys(), [curr_weights[key] - self.update_lr * gradients[key] for key in curr_weights.keys()]))

                #increment counter and return values
                return [a, b, la, lb, i+1, curr_weights, task_lossesb]

            def condition(a, b, la, lb, i, curr_weights, task_lossesb):
                return i < tf.reduce_sum(self.num_sgd_steps)

            def condition_numUpdates(a, b, la, lb, j, curr_weights, task_lossesb):
                return j < tf.reduce_sum(self.num_updates)-1

            def task_metalearn(inp, reuse=True):

                # init vars
                inputa, inputb, labela, labelb, = inp
                #############num_sgd_steps = self.num_sgd_steps
                #############num_updates = self.num_updates
                #task_outputbs=[]
                #task_lossesb=[]
                task_lossesb=tf.Variable([])

                fast_weights = weights
                k=self.config['update_batch_size']

                shapes={}
                for key in fast_weights.keys():
                    curr_shape = fast_weights[key].get_shape()
                    shapes.update({key:curr_shape})
                self.shapes = shapes

                ########THE 1st STEP JUST SO I CAN SAVE LOSS(f_theta0) on A
                temp_outp = self.forward(inputa[-k:], fast_weights, reuse=True, meta_loss=self.config['meta_loss'])  
                task_lossa = self.loss_func(temp_outp, labela[-k:])
                ################

                # loop where you do theta_i+1 = L(f_theta_i(inputa_chunk_i)), with chunks progressing by 1 point each time
                i=tf.constant(0)
                loop_result = tf.while_loop(condition, takeAStep, [inputa, inputb, labela, labelb, i, fast_weights, task_lossesb], shape_invariants=[inputa.get_shape(), inputb.get_shape(), labela.get_shape(), labelb.get_shape(), i.get_shape(), self.shapes, tf.TensorShape([None])])
                fast_weights=loop_result[5]

                '''for i in range(num_sgd_steps):
                    #f_theta_i(inputa_chunk_i)
                    task_outputa = self.forward(inputa[-k-num_sgd_steps+i:-num_sgd_steps+i], fast_weights, reuse=True, meta_loss=self.config['meta_loss'])  # only reuse on the first iter
                    #L(f_theta_i(inputa_chunk_i))
                    task_lossa = self.loss_func(task_outputa, labela[-k-num_sgd_steps+i:-num_sgd_steps+i])

                    # theta_i+1 = use L(f_theta_i(inputa_chunk_i))) to take a gradient step on theta_i
                    grads = tf.gradients(task_lossa, list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(
                        zip(fast_weights.keys(), [fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]))'''

                # f_thetaLast(inputb)
                output = self.forward(inputb, fast_weights, reuse=True, meta_loss=False)
                #task_outputbs.append(output)

                # L(f_thetaLast(inputb))
                #task_lossesb.append(self.loss_func(output, labelb))
                task_lossesb = tf.concat([task_lossesb, [self.loss_func(output, labelb)]], 0)

                # if taking more inner-update gradient steps (ie repeat the above sequential process)
                j=tf.constant(0)
                outer_loop_result = tf.while_loop(condition_numUpdates, innerLoop, [inputa, inputb, labela, labelb, j, fast_weights, task_lossesb], shape_invariants=[inputa.get_shape(), inputb.get_shape(), labela.get_shape(), labelb.get_shape(), j.get_shape(), shapes, tf.TensorShape([None])])
                fast_weights = outer_loop_result[5]
                #task_outputbs = outer_loop_result[7]
                task_lossesb = outer_loop_result[6]

                ################### not used anyway...
                task_outputa = 0.0 
                task_outputbs = 0.0

                '''for j in range(num_updates - 1):
                    for i in range(num_sgd_steps):
                        loss = self.loss_func(self.forward(inputa[-k-num_sgd_steps+i:-num_sgd_steps+i], fast_weights, reuse=True,
                                                           meta_loss=self.config['meta_loss']), labela[-k-num_sgd_steps+i:-num_sgd_steps+i]) 
                        grads = tf.gradients(loss, list(fast_weights.values()))
                        gradients = dict(zip(fast_weights.keys(), grads))
                        fast_weights = dict(
                            zip(fast_weights.keys(), [fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]))
                    # L(f_thetaLast(inputb))
                    output = self.forward(inputb, fast_weights, reuse=True, meta_loss=False)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))'''

                # task_outputa :    f_weights{0}(inputa)
                # task_outputbs :   [f_weights{1}(inputb), f_weights{2}(inputb), ...]
                # task_lossa :      loss(f_weights{0}(inputa))
                # task_lossesb :    [loss(f_weights{1}(inputb)), loss(f_weights{2}(inputb)), ...]
                #task_output = [task_outputa, task_outputbs, self.task_lossa, task_lossesb]
                task_output = (task_outputa, task_outputbs, task_lossa, task_lossesb)
                return task_output

            # to initialize the batch norm vars
            # might want to combine this, and not run idx 0 twice.
            if self.regressor.norm is not 'None':
                _ = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            #########################################################
            ## Output of performing inner gradient calculations
            #########################################################

            out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32]] #2nd thing should be [] but empty for now
            #result = tf.map_fn(task_metalearn, elems=(inputa, inputb, labela, labelb), dtype=out_dtype, parallel_iterations=self.config['meta_batch_size'])
            result = tf.map_fn(task_metalearn, elems=(inputa, inputb, labela, labelb), parallel_iterations=self.config['meta_batch_size'])
            outputas, outputbs, lossesa, self.lossesb = result

        ################################################
        ## Calculate preupdate and postupdate losses
        ################################################

        self.lossesb_transpose = tf.transpose(self.lossesb) #this is now numupdates x metaBS (total_losses2 should become dim = numupdates)

        def getMean(k, curr_list):
            temp = tf.gather(self.lossesb_transpose,k)
            num = tf.reduce_mean(temp)
            curr_list = tf.concat([curr_list, [num]], 0)
            return [k+1, curr_list]
        def conditionMean(k, curr_list):
            return k < self.num_updates

        # assign vars
        self.outputas, self.outputbs = outputas, outputbs

        # avg loss(f_weights{0}(inputa)) ####### TO DO check what this actually corresponds to now
        self.total_loss1 = total_loss1 = tf.reduce_mean(lossesa)

        # [avg loss(f_weights{1}(inputb)), avg loss(f_weights{2}(inputb)), ...] --> avg over metaBS
        #self.total_losses2 = [tf.reduce_mean(self.lossesb[j]) for j in range(self.num_updates)]
        k=tf.constant(0)
        curr_list=tf.Variable([])
        mean_loop_result = tf.while_loop(conditionMean, getMean, [k, curr_list], shape_invariants=[k.get_shape(), tf.TensorShape([None])])
        self.total_losses2 = total_losses2 = mean_loop_result[1]

        #define test op
        # self.test_op = tf.train.GradientDescentOptimizer(self.update_lr).minimize(total_loss1)

        #########################################
        ## Define pretrain_op
        #########################################

        #UPDATE weights{0} using loss(f_weights{0}(inputa))
        #standard supervised learning ... used to be used at test time, while performing rollout

        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
        k = self.config['update_batch_size']
        self.curr_weights = weights
        inputa = inputa[0]
        labela = labela[0]
        ##use past k steps to update weights w gradient descent
        task_outputa = self.forward(inputa[-k:], self.curr_weights, reuse=True, meta_loss=self.config['meta_loss'])
        # calculate loss(f_weights{0}(inputa))
        self.task_lossa_useLater = self.loss_func(task_outputa, labela[-k:])

        # update weights = use loss(f_weights{0}(inputa)) to take a gradient step
        grads = tf.gradients(self.task_lossa_useLater, list(self.curr_weights.values()))
        self.curr_gradients = dict(zip(self.curr_weights.keys(), grads))
        self.adapted_weights = dict(zip(self.curr_weights.keys(), [self.curr_weights[key] - self.update_lr * self.curr_gradients[key] for key in self.curr_weights.keys()]))

        #the operation
        self.test_op = [tf.assign(v, self.adapted_weights[k]) for k, v in weights.items()]

        #########################################
        ## Define metatrain_op
        #########################################

        #UPDATE theta_0 using loss(f_thetaLast_lastTime(inputb))

        optimizer = tf.train.AdamOptimizer(self.meta_lr)
        self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[-1])
        ####### self.gvs = gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs] ####### TO FIX??
        self.metatrain_op = optimizer.apply_gradients(gvs)

        ##############################################
        ## Summaries
        ##############################################

        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)

        #for j in range(self.num_updates):
        #    tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
        tf.summary.scalar(prefix+'Post-update loss, last step ', total_losses2[-1])


