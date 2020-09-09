import numpy as np
import torch
from .NPModel import NeuralProcessModel
from collections import deque
import random

def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

class NP(object):
    name = "NP"
    def __init__(self, NP_config, dataset=None):
        super().__init__()
        model_config = NP_config["model_config"]
        training_config = NP_config["training_config"]

        self.state_dim = model_config["state_dim"]
        self.action_dim = model_config["action_dim"]
        if "max_delay_step" in model_config:
            self.state_dim += model_config['max_delay_step']*self.action_dim
        self.x_dim = self.state_dim + self.action_dim
        self.y_dim = self.state_dim

        self.n_epochs = training_config["n_epochs"]
        self.lr = training_config["learning_rate"]
        self.batch_size = training_config["batch_size"]

        self.np_hidden_list = model_config['np_hidden_list']
        self.np_hidden_list_decoder = model_config['np_hidden_list_decoder']
        self.np_latent_dim = model_config['np_latent_dim']
        
        self.np_context_max_num = model_config['np_context_max_num']
        self.np_predict_context_num = model_config['np_predict_context_num']

        if model_config["load_model"]:
            self.model = CUDA(torch.load(model_config["model_path"]))
        else:
            self.model = CUDA(NeuralProcessModel(x_dim=self.x_dim,
                                                 y_dim=self.y_dim,
                                                 mlp_hidden_size_list=self.np_hidden_list,
                                                 mlp_hidden_size_list_decoder=self.np_hidden_list_decoder,
                                                 latent_dim=self.np_latent_dim,
                                                 use_rnn=False,
                                                 use_self_attention=True,
                                                 use_deter_path=True))

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.X = None
        self.Y = None
        self.memory = deque(maxlen=100)
    
    def reset(self):
        if self.X is not None:
            self.memory.append((self.X, self.Y))
            self.X = None
            self.Y = None
            
    def deep_reset(self):
        self.memory = deque(maxlen=1e3)
        self.X = None
        self.Y = None
            
    def data_process(self, data):
        s = data[1][None]
        a = data[2][None]
        label = data[3][None] # here label means the next state
        data = np.concatenate((s, a), axis=1)

        # add new data point to data buffer
        if self.X is None:
            self.X = CUDA(torch.Tensor(data))
            self.Y = CUDA(torch.Tensor(label))
        else:
            self.X = torch.cat((self.X, CUDA(torch.tensor(data).float())), dim=0)
            self.Y = torch.cat((self.Y, CUDA(torch.tensor(label).float())), dim=0)
    

    def train(self, data=None):
        if data is not None:
            self.data_process(data)
        
        training_data = random.sample(self.memory, k = min(self.batch_size, len(self.memory)))
                                      
        for batch in range(len(training_data)):
            
            (X, Y) = training_data[batch]
            data_num = len(X)

            X = X.view((1, -1, self.x_dim))  # (1, data_num, x_dim)
            Y = Y.view((1, -1, self.y_dim))  # (1, data_num, y_dim)

            self.model.train()

            for epoch in range(self.n_epochs):
                indices = list(range(data_num))
                np.random.shuffle(indices)
                num_context = np.random.randint(1,min(self.np_context_max_num,data_num))
                num_target = num_context + np.random.randint(0,min(self.np_context_max_num,data_num)-num_context)
                rand_ind_ctt, rand_ind_tgt = indices[:num_context], indices[:num_target]
                context_x = X[:, rand_ind_ctt, :]
                context_y = Y[:, rand_ind_ctt, :]
                target_x = X[:, rand_ind_tgt, :]
                target_y = Y[:, rand_ind_tgt, :]
                self.optim.zero_grad()
                mu, sigma, log_p, kl, loss = self.model(context_x, context_y, target_x, target_y)
                loss.backward()
                self.optim.step()
        return loss.item()

    def predict(self, s, a):
        # convert to torch format
        self.model.eval()
        s = CUDA(torch.tensor(s).float())
        a = CUDA(torch.tensor(a).float())
        inputs = torch.cat((s, a), axis=1) 
        target_x = inputs.view((1, -1, self.x_dim))

        data_num = len(self.X)
        indices = list(range(data_num))
        np.random.shuffle(indices)
        indices = indices[:self.np_predict_context_num]
        
        exist_x_tensor = self.X[indices, :]
        exist_y_tensor = self.Y[indices, :]

        context_x = CUDA(exist_x_tensor.view(1, -1, self.x_dim))
        context_y = CUDA(exist_y_tensor.view(1, -1, self.y_dim))
        
        print_latent = True

        mu, sigma, log_p, kl, loss = self.model(context_x, context_y, target_x, None, print_latent)

        mu = torch.squeeze(mu, 0)
        sigma = torch.squeeze(sigma, 0)
        return mu.cpu().detach().numpy()