'''
@Author: Jiacheng Zhu
@Email:
@Date:
@LastEditTime:
@Description:
'''

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from loguru import logger
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from ANP.NPModel import NeuralProcessModel


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


class NPComponent(object):
    # output: [state mean, state var]
    name = "NP"

    def __init__(self, NP_config, dataset=None):
        super().__init__()
        model_config = NP_config["model_config"]
        training_config = NP_config["training_config"]

        self.state_dim = model_config["state_dim"]
        self.action_dim = model_config["action_dim"]
        self.x_dim = self.state_dim + self.action_dim
        self.y_dim = model_config["state_dim"]

        self.n_epochs = training_config["n_epochs"]
        self.lr = training_config["learning_rate"]
        self.batch_size = training_config["batch_size"]

        self.save_model_flag = training_config["save_model_flag"]
        self.save_model_path = training_config["save_model_path"]

        self.validation_flag = training_config["validation_flag"]
        self.validate_freq = training_config["validation_freq"]
        self.validation_ratio = training_config["validation_ratio"]

        # TODO: Let NP use the config file
        # NOTICE: NP configuration

        # NOTICE: 'gp' use all existing data as context, input data as target
        #           (do the inference conditioned on previous data)
        #         'nn' use input data as both context and target
        #           (NP learns the "function" rather than memorizing data)
        # print('model_config["likelihood_method"] =', model_config["likelihood_method"])
        self.likelihood_method = model_config["likelihood_method"]  # 'gp' or 'nn'
        if self.likelihood_method not in ['gp', 'nn']:
            print("Please select inference method, 'gp' or 'nn'!")

        # NOTICE: 'loss' use the negative loss to decide the 'likelihood' F(data, model)
        #          'll' use the sum of the likelihood of F(data, model) = N(data , pred_mu, pred_var)
        self.likelihood_value = model_config['likelihood_value']   # 'll' or 'loss'
        if self.likelihood_value not in ['ll', 'loss']:
            print("Please select likelihood value, 'loss' or 'll'!")

        self.sequential = model_config['sequential']
        self.virtual_batch = model_config['virtual_batch']
        self.np_hidden_list = model_config['np_hidden_list']
        self.np_latent_dim = model_config['np_latent_dim']

        if model_config["load_model"]:
            self.model = CUDA(torch.load(model_config["model_path"]))
        else:
            self.model = CUDA(NeuralProcessModel(x_dim=self.x_dim,
                                                 y_dim=self.y_dim,
                                                 mlp_hidden_size_list=self.np_hidden_list,
                                                 latent_dim=self.np_latent_dim,
                                                 use_rnn=False,
                                                 use_self_attention=True,
                                                 use_deter_path=True))

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if dataset:
            self.dataset = dataset
        else:
            self.dataset = []

    # NOTICE: Fit
    def fit(self, dataset=None, logger=True):
        '''
        :param dataset: list data
        :param logger:
        :return:
        '''
        # NOTICE: tunable parameter or methods
        #   self.epoch
        #   self.sequential = True or False
        #   self.virtual_batch = False
        #
        print('fitting model')
        if dataset is not None:
            data_list = dataset
        else:
            data_list = self.dataset

        # print('data_list =', data_list)

        # convert list of data into X and Y tensor
        x_tensor, y_tensor = self.list_2_torch(data_list)  # (data_num, x_dim), (data_num, y_dim)

        context_x = CUDA(x_tensor.view((1, -1, self.x_dim)))  # (1, data_num, x_dim)
        context_y = CUDA(y_tensor.view((1, -1, self.y_dim)))  # (1, data_num, y_dim)
        target_x = context_x  # (1, data_num, x_dim)
        target_y = context_y  # (1, data_num, y_dim)

        data_num = len(data_list)  # obtain the num of data (sequence length)

        self.model.train()
        for epoch in range(self.n_epochs):

            # NOTICE: do not consider the sequential order in data, then
            #   randomly permutate them
            if not self.sequential:
                # NOTICE: permutate X and Y respectively in each step
                rand_ind_ctt = torch.randperm(data_num)
                rand_ind_tgt = torch.randperm(data_num)

                context_x = context_x[:, rand_ind_ctt, :]
                context_y = context_y[:, rand_ind_ctt, :]
                target_x = target_x[:, rand_ind_tgt, :]
                target_y = target_y[:, rand_ind_tgt, :]

            # NOTICE: forward
            self.optim.zero_grad()
            mu, sigma, log_p, kl, loss = self.model(context_x, context_y,
                                                    target_x, target_y)
            loss.backward()
            self.optim.step()
            if logger:
                print(f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss:.4f}.")
        return loss

    def fit_prior(self, dataset=None, logger=True):
        '''
        :param dataset: list data
        :param logger:
        :return:
        '''
        print('Fitting prior NP (using small number of iterations)')
        # NOTICE: tunable parameter or methods
        #   self.epoch
        #   self.sequential = True or False
        #   self.virtual_batch = False
        #
        if dataset is not None:
            data_list = dataset
        else:
            data_list = self.dataset
        data_num = len(data_list)  # obtain the num of data (sequence length)

        # convert list of data into X and Y tensor
        x_tensor, y_tensor = self.list_2_torch(dataset)  # (data_num, x_dim), (data_num, y_dim)
        context_x = CUDA(x_tensor.view((1, -1, self.x_dim)))  # (1, data_num, x_dim)
        context_y = CUDA(y_tensor.view((1, -1, self.y_dim)))  # (1, data_num, y_dim)
        target_x = context_x  # (1, data_num, x_dim)
        target_y = context_y  # (1, data_num, y_dim)

        # NOTICE: prior epoch
        self.model.train()
        for epoch in range(300):

            # NOTICE: do not consider the sequential order in data, then
            #   randomly permutate them
            if not self.sequential:
                # NOTICE: permutate X and Y respectively in each step
                rand_ind_ctt = torch.randperm(data_num)
                rand_ind_tgt = torch.randperm(data_num)

                context_x = context_x[:, rand_ind_ctt, :]
                context_y = context_y[:, rand_ind_ctt, :]
                target_x = target_x[:, rand_ind_tgt, :]
                target_y = target_y[:, rand_ind_tgt, :]

            # NOTICE: forward
            self.optim.zero_grad()
            mu, sigma, log_p, kl, loss = self.model(context_x, context_y,
                                                    target_x, target_y)
            loss.backward()
            self.optim.step()
            if logger:
                print(f"training epoch [{epoch}/{self.n_epochs}],loss train: {loss:.4f}.")
        return loss

    # NOTICE: Completely rewrite it
    def likelihood(self, dataset):
        self.model.eval()
        # dataset : list of [task_idx, state, action, next_state-state]
        # input_data_num = len(dataset)  # obtain the num of data (sequence length)

        # convert list of data into X and Y tensor
        input_x_tensor, input_y_tensor = self.list_2_torch(dataset)  # (input_data_num, x_dim), (input_data_num, y_dim)
        target_x = CUDA(input_x_tensor.view((1, -1, self.x_dim)))  # (1, input_data_num, x_dim)
        target_y = CUDA(input_y_tensor.view((1, -1, self.y_dim)))  # (1, input_data_num, y_dim)

        if self.likelihood_method == 'gp':
            # NOTICE: Use all the existing data as context
            # exist_data_num = len(self.dataset)
            exist_x_tensor, exist_y_tensor = self.list_2_torch(self.dataset)
            context_x = CUDA(exist_x_tensor.view(1, -1, self.x_dim))
            context_y = CUDA(exist_y_tensor.view(1, -1, self.y_dim))
        elif self.likelihood_method == 'nn':
            context_x = target_x
            context_y = target_y
        else:
            print('Please choose likelihood method')

        # NOTICE: forward the NP
        mu, sigma, log_p, kl, loss = self.model(context_x, context_y, target_x, target_y)

        if self.likelihood_value == 'loss':
            # NOTICE: Use negative loss to represent the likelihood F(input_data, model)
            return - loss.cpu().detach().numpy()

        elif self.likelihood_value == 'll':
            # NOTICE: Use the sum of the likelihood of N(data | pred_mu, pred_var)
            mu_dist_v = mu.view(-1, self.y_dim)  # (test_num, y_dim)
            sigma_dist_v = sigma.view(-1, self.y_dim)  # (test_num, y_dim)
            cov = torch.diag_embed(sigma_dist_v)  # (test_num, y_dim, y_dim)
            mg = torch.distributions.MultivariateNormal(mu_dist_v, cov)
            ll = mg.log_prob(target_y.view((-1, self.y_dim)))
            ll_sum = torch.sum(ll)
            return ll_sum.cpu().detach().numpy()

    # TODO: Add loglikelihood()?
    #   it is only used in svit
    def log_likelihood(self, data_point):
        return

    # NOTICE, convert list data into tensor for NP
    def list_2_torch(self, data_list):
        # print('data_list =', data_list)
        '''
        :param data_list: [[index, array([ current_state ], array([ action ]), array([ next_state ])],
                           [index, array([ current_state ], array([ action ]), array([ next_state ])],
                           ...
                           [index, array([ current_state ], array([ action ]), array([ next_state ])]]
        :return:    tensor(data_num, x_dim), tensor(data_num, y_dim)
        '''
        # print('list_2_torch data_list=', data_list)
        # print('len(self.dataset) =', len(self.dataset))
        data_len = len(data_list)
        ctt_x_array = np.zeros((data_len, self.x_dim))
        ctt_y_array = np.zeros((data_len, self.y_dim))
        for i, data in enumerate(data_list):
            cluster_id, current_env, action, next_env = data
            ctt_x_array[i, 0:self.state_dim] = current_env
            ctt_x_array[i, self.state_dim: self.state_dim + self.action_dim] = action
            ctt_y_array[i, :] = next_env
        return torch.Tensor(ctt_x_array), torch.Tensor(ctt_y_array)

    # NOTICE: Predict
    # TODO: forward the NP to make the prediction
    def predict(self, s, a):
        # convert to torch format
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1)  # (1, x_dim)
        target_x = inputs.view((1, 1, self.x_dim))

        # NOTICE: Forward NP to generate target_y
        #  use the existing data as context
        exist_x_tensor, exist_y_tensor = self.list_2_torch(self.dataset)
        context_x = CUDA(exist_x_tensor.view(1, -1, self.x_dim))
        context_y = CUDA(exist_y_tensor.view(1, -1, self.y_dim))

        # NOTICE: forward the NP
        mu, sigma, log_p, kl, loss = self.model(context_x, context_y, target_x, None)

        # TODO: not sure about the output dim of mu and sigma
        mu = torch.squeeze(mu, 0)
        sigma = torch.squeeze(sigma, 0)
        return mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()

    # NOTICE: Canonical function
    def add_data_point(self, data):
        # data format: [task_idx, state, action, next_state-state]
        self.dataset.append(data)

    # NOTICE: Canonical function
    def reset_dataset(self, new_dataset=None):
        # dataset format: list of [task_idx, state, action, next_state-state]
        if new_dataset is not None:
            self.dataset = new_dataset
        else:
            self.dataset = []


if __name__ == '__main__':
    model = NPComponent()