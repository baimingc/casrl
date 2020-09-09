'''
@Author: Wenhao Ding
@Email: 
@Date: 2020-03-17 12:32:22
@LastEditTime: 2020-04-04 17:19:54
@Description: 
'''

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive, LessThan

import collections
import numpy as np


#gpytorch.settings.max_eager_kernel_size(1000)
#gpytorch.settings.max_cholesky_size(1000)


def CUDA(var):
    return var
    #return var.cuda() if torch.cuda.is_available() else var


class ExactGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, params):
        super(ExactGPR, self).__init__(train_x, train_y, likelihood)

        self.lengthscale_prior = None #gpytorch.priors.GammaPrior(3.0, 6.0)
        self.outputscale_prior = None #gpytorch.priors.GammaPrior(2.0, 0.15)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim, 
                lengthscale_prior=self.lengthscale_prior,
                lengthscale_constraint=LessThan(params[0])
            ),
            outputscale_prior=self.outputscale_prior,
            outputscale_constraint=GreaterThan(params[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset_parameters(self, params):
        self.likelihood.noise_covar.initialize(noise=params[0])
        self.mean_module.initialize(constant=params[1])
        self.covar_module.initialize(outputscale=params[2])
        self.covar_module.base_kernel.initialize(lengthscale=params[3:8])
        #self.covar_module.base_kernel.lengthscale = self.lengthscale_prior.mean
        #self.covar_module.outputscale = self.outputscale_prior.mean


class GPComponent(object):
    def __init__(self):
        self.data = None
        self.n = 0

        self.lr = 0.1
        self.state_dim = 4
        self.action_dim = 1
        self.input_dim = self.state_dim + self.action_dim
        self.gp_iter = 10

        # constraint for scalekernel, avoid getting a very small variance
        # prior of the kernel parameters
        # [NOTE] these prior parameters should be similar to the estimated parameters of real data
        # if lengthscale is too large, it will be too difficult to create new components
        # if lengthscale is too small, it will be too esay to create new components
        # if noise_covar is too large, the prediction will be inaccurate
        # if noise_covar is too small, the covariance will be very small, causing some numerical problems
        lengthscale = 1
        self.param = [
            1e-5,   # noise_covar initilize and constraint
            0.0,    # constant initilize
            0.1,    # outputscale initilize
            lengthscale, lengthscale, lengthscale, lengthscale, lengthscale, # [lengthscale initilize]
            100.0,  # lengthscale_constraint
            0.0000001   # outputscale_constraint
        ] 
        self.param = CUDA(torch.tensor(self.param))

        # initialize model and likelihood
        model_list = []
        likelihood_list = []
        for m_i in range(self.state_dim):
            likelihood = CUDA(gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(self.param[0])))
            model = CUDA(ExactGPR(None, None, likelihood, self.input_dim, self.param[8:10]))
            model.reset_parameters(self.param)
            likelihood_list.append(model.likelihood)
            model_list.append(model)

        # initialize model list
        self.model = gpytorch.models.IndependentModelList(*model_list)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)

        # initialize optimizer
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)

        # change the flag
        self.model.eval()
        self.likelihood.eval()

    def reset_parameters(self):
        for m_i in range(self.state_dim):
            self.model.models[m_i].reset_parameters(self.param)

    def reset_point_for_test(self, test_data):
        # reset previous parameters
        self.reset_parameters()

        '''
        # a placeholder for model creation, this data will be replaced when this component is trained.
        data_placeholder = CUDA(torch.Tensor(test_data))
        train_x = data_placeholder[:, :self.input_dim]
        train_y = data_placeholder[:, self.input_dim:]
        for m_i in range(self.state_dim):
            self.model.models[m_i].set_train_data(train_x, train_y[:, m_i], strict=False)
        # DO NOT add 1 to n, we dont want to train the model
        #self.n = 1

        # train the new model with one test data
        self.model.train() # set prediction_strategy = None inside
        self.likelihood.train()
        for i in range(self.gp_iter):
            self.optimizer.zero_grad()
            output_func = self.model(*self.model.train_inputs)
            loss = -self.mll(output_func, self.model.train_targets)
            loss.backward()
            self.optimizer.step()
        self.model.eval()
        self.likelihood.eval()
        '''

    def train_model(self):
        # prepare data
        train_x = self.data[:, :self.input_dim]
        train_y = self.data[:, self.input_dim:]

        # reset training data
        # DO NOT reset parameters, use the parameters of last time
        for m_i in range(self.state_dim):
            self.model.models[m_i].set_train_data(train_x, train_y[:, m_i], strict=False)
            #self.model.models[m_i].prediction_strategy = None

        with gpytorch.settings.max_cg_iterations(1000):
            # training stage
            self.model.train() # set prediction_strategy = None inside
            self.likelihood.train()
            for i in range(self.gp_iter):
                self.optimizer.zero_grad()
                output_func = self.model(*self.model.train_inputs)
                loss = -self.mll(output_func, self.model.train_targets)

                print('Iter {}/{} - Loss: {}  outputscale: {}  lengthscale: {}   noise: {}'.format(
                    i + 1, 
                    self.gp_iter, 
                    loss.item(),
                    self.model.models[0].covar_module.outputscale.detach().numpy(),
                    self.model.models[0].covar_module.base_kernel.lengthscale.detach().numpy(),
                    self.model.models[0].likelihood.noise.detach().numpy()
                ))
                loss.backward()
                self.optimizer.step()

            # change the flag
            self.model.eval()
            self.likelihood.eval()

    def get_point(self):
        return self.data.cpu().numpy()

    def merge_point(self, new_tensor_data, new_list):
        # the data to be merged is expected to be a torch.Tensor
        self.data = torch.cat((self.data, new_tensor_data), dim=0)
        self.index_list += new_list
        self.n += len(new_list)

    def add_point(self, x):
        # add some noise to the data
        #x_noise = np.random.randn(1, 4) * np.sqrt(0.0001)
        #x[:, 5:9] = x[:, 5:9] + x_noise

        if self.data is None:
            self.data = CUDA(torch.Tensor(x))
        else:
            self.data = torch.cat((self.data, CUDA(torch.tensor(x).float())), dim=0)
        self.n += 1

    def del_point(self, x, i):
        # for sequential vi method, this function is deprecated
        # TODO: check this may be really slow, modify with index later
        remove_index = self.index_list.index(i)
        self.data = torch.cat([self.data[:remove_index,:], self.data[remove_index+1:,:]], dim=0)
        self.index_list.remove(i)
        self.n -= 1
        return self.n

    def log_posterior_pdf(self, x, train=True):
        # prepare data
        x = CUDA(torch.Tensor(x))
        test_x = x[:, :self.input_dim]
        test_y = x[:, self.input_dim:]

        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # [TODO] if likelihood is added here, the new component is hard to create
            sample_func = self.model(test_x, test_x, test_x, test_x)
            sample_func_lik = self.likelihood(*sample_func)

            log_ppf = 0
            for f_i in range(len(sample_func)):
                # [BUG of GPytorch] when numerical problem happens, the covariance_matrix will be non-positive-definite
                # then, the log_porb will return nan. We reset the covariance_matrix to a pre-defined value (constraint of noise_covar)
                if sample_func_lik[f_i].covariance_matrix <= CUDA(torch.tensor([[0.0]])):
                    sample_func_lik[f_i] = gpytorch.distributions.MultivariateNormal(sample_func_lik[f_i].loc, CUDA(self.param[0][None, None]))

                incre = sample_func_lik[f_i].log_prob(test_y[:, f_i]).item()
                log_ppf += incre

                if np.isnan(incre):
                    print('---------------NaN detected---------------')
                    print('f_i: ', f_i)
                    print('x', test_x)
                    print('y', test_y)
                    print('y[f_i] ', test_y[:, f_i])
                    print('loc: ', sample_func[f_i].loc.numpy())
                    print('covariance_matrix: ', sample_func[f_i].covariance_matrix.detach().cpu().numpy())
                    print('likelihood.covariance_matrix: ', sample_func[f_i].covariance_matrix.detach().cpu().numpy())
                    #print('likelihood.noise_covar.raw_noise', self.model.models[f_i].likelihood.noise_covar.raw_noise)
                    #print(self.model.models[f_i].state_dict())
                    print('lengthscale', self.model.models[f_i].covar_module.base_kernel.lengthscale)
                    print('lengthscale', self.model.models[f_i].covar_module.outputscale)
                    print('------------------------------------------')
                else:
                    print('lengthscale', self.model.models[f_i].covar_module.outputscale)
                    print('lengthscale', self.model.models[f_i].covar_module.base_kernel.lengthscale)
                    #print('covariance_matrix: ', sample_func[f_i].covariance_matrix.detach().cpu().numpy())
                    #print('likelihood.covariance_matrix: ', sample_func_lik[f_i].covariance_matrix.detach().cpu().numpy())
                #    #print(self.model.models[f_i].state_dict())
                #    print('------------------------------------------')
        # since we added all likelihood together
        return log_ppf/len(sample_func_lik)

    def predict(self, x):
        # prepare data
        test_x = CUDA(torch.Tensor(x[:, :self.input_dim]))

        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            sample_func = self.model(test_x, test_x, test_x, test_x)
            sample_func_lik = self.likelihood(*sample_func)
        return sample_func_lik



data = np.load('./gym_data.npy')[:, 0, :]
print(data.shape)
gp = GPComponent()
#gp = GPComponent(data[0:128])

#gpytorch.distributions.MultivariateNormal(torch.tensor([0]), torch.tensor([[-16.6508]]))

'''
param = collections.OrderedDict([
    ('likelihood.noise_covar.raw_noise', torch.tensor([-16.6508])), 
    ('mean_module.constant', torch.tensor([-0.0131])), 
    ('covar_module.raw_outputscale', torch.tensor(-0.4362)), 
    ('covar_module.base_kernel.raw_lengthscale', torch.tensor([[65.0337, 56.3517,  2.6550, 56.5269,  5.8988]]))
])  
gp.model.models[3].load_state_dict(param)
'''

for i in range(801):
    gp.add_point(data[i, :][None])
    gp.train_model()


#for i in range(801, 802):
for i in range(801, 804):
    log_ppf = gp.log_posterior_pdf(data[i, :][None])
    print(log_ppf)
