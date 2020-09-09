'''
@Author: Mengdi Xu, Wenhao Ding
@Email: 
@Date: 2020-03-17 12:32:22
@LastEditTime: 2020-03-18 17:36:58
@Description: 
'''

import gpflow
from gpflow.utilities import print_summary, set_trainable, to_default_float
import tensorflow_probability as tfp
import numpy as np
from loguru import logger


class GPComponent(object):
    def __init__(self, hyperparam, data, index_set):
        self.n = 0
        self.data = data
        self.index_list = index_set

        # data for GPR
        self.X = self.data[:, 1:3]
        self.Y = self.data[:, 3]

        # define kernels
        self.hyperparam = hyperparam
        self.k1 = gpflow.kernels.SquaredExponential(active_dims=[0])
        self.k2 = gpflow.kernels.SquaredExponential(active_dims=[1])

        # # set the prior of the parameters
        # self.k1.variance.prior = tfp.distributions.Gamma(
        #     to_default_float(hyperparam[0]), to_default_float(hyperparam[1]))
        # self.k1.lengthscale.prior = tfp.distributions.Gamma(
        #     to_default_float(hyperparam[2]), to_default_float(hyperparam[3]))
        # self.k2.variance.prior = tfp.distributions.Gamma(
        #     to_default_float(hyperparam[4]), to_default_float(hyperparam[5]))
        # self.k2.lengthscale.prior = tfp.distributions.Gamma(
        #     to_default_float(hyperparam[6]), to_default_float(hyperparam[7]))

        # initialize model and optimizer
        self.opt = gpflow.optimizers.Scipy()
        self.model = gpflow.models.GPR(data=(self.X, self.Y), kernel=self.k1+self.k2, mean_function=None)

    def objective_closure(self):
        return -self.model.log_marginal_likelihood()

    def train_model(self, max_iteration=100):
        opt_logs = self.opt.minimize(self.objective_closure, self.model.trainable_variables, options=dict(maxiter=max_iteration))
        return opt_logs

    def add_point(self, x, i):
        self.data = np.vstack((self.data, x))
        self.index_list.append(i)
        self.n += 1

        # create a new GPR model with new data
        self.X = self.data[:, 1:3]
        self.Y = self.data[:, 3]
        self.model = gpflow.models.GPR(data=(self.X, self.Y), kernel=self.k1+self.k2, mean_function=None)

    def del_point(self, x, i):
        # TODO: check this may be really slow, modify with index later
        remove_index = self.index_list.index(i)
        self.data = np.delete(self.data, remove_index, 0)
        self.index_list.remove(i)
        self.n -= 1
        return self.n

    def log_posterior_pdf(self, x):
        if self.n == 0:
            # a new cluster just initialized
            # sample real parameters
            param = []
            for param_index in range(4): # currently only 4 parameters
                s_ind = param_index*2
                dist = tfp.distributions.Gamma(self.hyperparam[s_ind], self.hyperparam[s_ind+1])
                param.append(dist.sample())
            
            # assign kernel parameters
            self.k1.variance.assign(param[0])
            self.k1.lengthscales.assign(param[1])
            self.k2.variance.assign(param[2])
            self.k2.lengthscales.assign(param[3])
        else:
            #logger.info("Start train")
            self.train_model(max_iteration=10)
            #logger.info("Finish train")

        # get log-likelihood
        logp = np.array(self.model.predict_log_density((x[:,1:3], x[:,3])))
        return logp[0][0]
