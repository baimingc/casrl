'''
@Author: Mengdi Xu
@Email: 
@Date: 2020-03-17 12:32:22
@LastEditTime: 2020-03-18 23:26:29
@Description: 
'''

import numpy as np

import torch

import pyro
import pyro.contrib.gp as gp
from pyro.distributions import MultivariateNormal, Gamma


class GPComponent(object):
    def __init__(self, hyperparam, data, index_list):
        self.lr = 0.01
        self.n = 0
        self.data = data
        self.index_list = index_list
        self.hyperparam = hyperparam

        # initialize the model
        self.k1 = []
        self.k2 = []
        self.model = []
        self.optimizer = []

        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        
    def train_model(self, max_iteration=100):
        # prepare data
        train_x = torch.Tensor(self.data[:, 1:3])
        train_y = torch.Tensor(self.data[:, 3])
        self.model.set_data(train_x, train_y)

        # training stage
        for i in range(max_iteration):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model.model, self.model.guide)

            loss.backward()
            self.optimizer.step()

    def add_point(self, x, i):
        self.data = np.vstack((self.data, x))
        self.index_list.append(i)
        self.n += 1

    def del_point(self, x, i):
        # TODO: check this may be really slow, modify with index later
        remove_index = self.index_list.index(i)
        self.data = np.delete(self.data, remove_index, 0)
        self.index_list.remove(i)
        self.n -= 1
        return self.n

    def log_posterior_pdf(self, x):
        # prepare data
        test_x = torch.Tensor(x[:, 1:3])
        test_y = torch.Tensor(x[:, 3])

        if self.n == 0:  # a new cluster just initialized
            # prior of the kernel parameters
            train_x = torch.Tensor(self.data[:, 1:3])
            train_y = torch.Tensor(self.data[:, 3])

            self.k1 = gp.kernels.RBF(input_dim=2, active_dims=[True, False])
            self.k1.variance = Gamma(self.hyperparam[0], self.hyperparam[1]).sample()
            self.k1.lengthscale = Gamma(self.hyperparam[2], self.hyperparam[3]).sample()

            self.k2 = gp.kernels.RBF(input_dim=2, active_dims=[False, True])
            self.k2.variance = Gamma(self.hyperparam[4], self.hyperparam[5]).sample()
            self.k2.lengthscale = Gamma(self.hyperparam[6], self.hyperparam[7]).sample()

            self.kernel = gp.kernels.Sum(self.k1, self.k2)
            
            self.model = gp.models.GPRegression(train_x, train_y, self.kernel, noise=None)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            # train model
            self.train_model(max_iteration=3)
            #print('-----')

        # get the log likelihood
        with torch.no_grad():
            mean, cov = self.model(test_x, full_cov=True, noiseless=False)
            logp = MultivariateNormal(mean, cov).log_prob(test_y).item() 
        return logp
