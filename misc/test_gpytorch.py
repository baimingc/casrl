'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-03-18 18:50:25
@LastEditTime: 2020-05-10 22:11:52
@Description: 
'''

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
# when noise is 0, the loss will continue decreasing
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.1)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)


# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 500


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    '''
    print('Iter %d/%d - Loss: %.3f  outputscale: %.3f  lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.outputscale.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    '''
    print('Iter %d/%d - Loss: %.3f lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()


model.eval()
likelihood.eval()

test_x = torch.linspace(0, 1, 100)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for i in range(100):
        test_x_i = test_x[i:i+1]
        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        observed_pred = likelihood(model(test_x_i))
        #print(observed_pred.covariance_matrix.detach().numpy())
        #print(model(test_x).covariance_matrix)
        #print(observed_pred.covariance_matrix)
