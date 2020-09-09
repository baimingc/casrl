'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-03-17 12:32:22
@LastEditTime: 2020-03-22 18:57:21
@Description: 
'''

import numpy as np
from loguru import logger

import torch
import gpytorch


def CUDA(var):
    return var
    #return var.cuda() if torch.cuda.is_available() else var


class ExactGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module_1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=0))
        self.covar_module_2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module_1(x) + self.covar_module_2(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def reset_parameters(self, params):
        self.covar_module_1.raw_outputscale.data.copy_(params[0])
        self.covar_module_1.base_kernel.raw_lengthscale.data.copy_(params[1])
        self.covar_module_2.raw_outputscale.data.copy_(params[2])
        self.covar_module_2.base_kernel.raw_lengthscale.data.copy_(params[3])


class GPComponent(object):
    def __init__(self, hyperparam, data, index_list):
        self.lr = 0.1
        self.n = 0
        self.data = data
        self.index_list = index_list
        self.hyperparam = hyperparam

        # prepare data
        train_x = CUDA(torch.Tensor(self.data[:, 1:3]))
        train_y = CUDA(torch.Tensor(self.data[:, 3]))

        # initialize model and likelihood
        self.likelihood = CUDA(gpytorch.likelihoods.GaussianLikelihood())
        self.model = CUDA(ExactGPR(train_x, train_y, self.likelihood))

        # initialize optimizer
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        self.mll = CUDA(gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model))

        # prior of the kernel parameters
        self.param = []
        for param_index in range(4): # currently only 4 parameters
            s_ind = param_index*2
            dist = CUDA(torch.distributions.Gamma(self.hyperparam[s_ind], self.hyperparam[s_ind+1]))
            self.param.append(dist.sample())
        self.model.reset_parameters(self.param.copy())

    def train_model(self, max_iteration=100):
        # prepare data
        train_x = CUDA(torch.Tensor(self.data[:, 1:3]))
        train_y = CUDA(torch.Tensor(self.data[:, 3]))

        # reset training data and parameters
        # parameters must be reset since the data has changed a lot
        self.model.set_train_data(train_x, train_y, strict=False)
        self.model.reset_parameters(self.param.copy())

        # training stage
        self.model.train()
        self.likelihood.train()
        for i in range(max_iteration):
            self.optimizer.zero_grad()
            output_func = self.model(train_x)
            loss = - torch.mean(self.mll(output_func, train_y))
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
        # not the first data
        if self.n > 0:  
            self.train_model(max_iteration=10)

        # prepare data
        test_x = CUDA(torch.Tensor(x[:, 1:3]))
        test_y = CUDA(torch.Tensor(x[:, 3]))

        # get the log likelihood
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            sample_func = self.model(test_x)
            log_ppf = sample_func.log_prob(test_y).item() 
        return log_ppf
