import numpy as np
from matplotlib import pyplot as plt
import collections
from tqdm.auto import tqdm
import torch
from NPModel import NeuralProcessModel
# from Sin_Wave_Data import sin_wave_data, plot_functions
import pickle
import time


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var

# NOTICE:
#   Ite = 1000 train precentage 0.1
#   train use all context with random perm order
#   mlp [1024, 1024, 1024, 1024], latent_dim = 1024
#   Both loss and  ll_mean, ll_sum finish the classfication with 30% training
#   training(from zero) time (GPU) ~0.08s*data_num in each episode
#   perfect classification


class NP_component_test():
    def __init__(self):
        self.iteration = 1500
        # Intereting,
        self.NP = CUDA(NeuralProcessModel(x_dim=6,
                                     y_dim=5,
                                     mlp_hidden_size_list=[1024, 1024, 1024, 1024],
                                     latent_dim=1024,
                                     use_rnn=False,
                                     use_self_attention=True,
                                     use_deter_path=True))
        self.optim = torch.optim.Adam(self.NP.parameters(), lr=2e-4)
        # self.data = data


    def fit(self, dataset, batch_size=1, print_info=False):
        train_start_time = time.time()
        self.NP.train()
        # TODO: put data in it
        # print('len(dataset) =', len(dataset))   # 1117
        # print('dataset[0] =', dataset[0])
        # NOTICE: Convert list data to torch for NP
        self.dataset = dataset  # Store the trained data
        x, y = list_2_torch(dataset)
        # (data_num, x_dim), (data_num, y_dim)
        # NOTICE: The there is no batch_size or sequence length here
        #   .
        #   Image NP as gaussian process it should be (B=1, seq_len, x_dim)
        #   Image NP as a neural network it should be (B=?, seq_len=?, x_dim)
        #   Here I use it as a neural network
        #   The "batch_size"
        data_num = x.size()[0]
        # print(data_num)
        x = x.view((data_num, 1, -1))   # (data_num, 1, x_dim)
        # print('1. context_x.size() =', context_x.size())
        y = y.view((data_num, 1, -1))   # (data_num, 1, y_dim)
        # print('context_y.size() =', context_y.size())
        for it in range(self.iteration):
            if print_info:
                print('it =', it)
            # random permutation
            rand_ind_ctt = torch.randperm(data_num)
            rand_ind_tgt = torch.randperm(data_num)
            # print('rand_indx =', rand_indx)
            context_x = x[rand_ind_ctt, :, :]
            context_y = y[rand_ind_ctt, :, :]

            target_x = x[rand_ind_tgt, :, :]
            target_y = y[rand_ind_tgt, :, :]

            # NOTICE: Here the input size is (batch_size, 1, x/y_dim)
            #   it is different from how we evaluate the model
            #   ignore the sequential info, use the NP as a NN
            #   *This one is better*
            context_x_i = context_x#[0:batch_size, :, :]
            context_y_i = context_y#[0:batch_size, :, :]
            target_x_i = target_x#[0:batch_size, :, :]
            target_y_i = target_y#[0:batch_size, :, :]

            # NOTICE: Another way, input size (1, seq_len, x/y_dim)
            #   then there is no "batch_size", Or batch must be 1 for each episode
            #   the "batch_size" of the training is the max_training data number
            #   (Can) consider the sequential, use the NP as GP
            #   In that case, the order should be kept
            # They have different result. when using loss to classify episode
            context_x_i = CUDA(context_x_i.view((1, -1, 6)))
            context_y_i = CUDA(context_y_i.view((1, -1, 5)))
            target_x_i = CUDA(target_x_i.view((1, -1, 6)))
            target_y_i = CUDA(target_y_i.view((1, -1, 5)))
            # NOTICE: (1, seq_len, x/y_dim)

            self.optim.zero_grad()
            # NOTICE: When training use (context_x, context_y) = (target_x, target_y)
            #   KL will always be 0
            mu, sigma, log_p, kl, loss = self.NP(context_x_i, context_y_i,
                                                 target_x_i, target_y_i)
            loss.backward()
            self.optim.step()

        training_time = time.time() - train_start_time
        print('training_time =', training_time)
            # print()
            # print('iteration = ', it, 'loss = ', loss)
            # print()
            # print('torch.mean(log_p) =', torch.mean(log_p))
            # print()


    def likelihood_np(self, test_dataset):
        self.NP.eval()
        # NOTICE: Compute the likelihood of data
        #   There are two ways
        #   (1) Output the loss of NP using
        #   context = existing data, target = test data
        #   (2) Use the output mean and variance and then
        #   compute the likelihood (like NN)

        # NOTICE: get raw data tensor
        test_x, test_y = list_2_torch(test_dataset)
        test_data_num = test_x.size()[0]
        test_x = test_x.view((test_data_num, 1, -1))  # (data_num, 1, x_dim)
        # print('1. context_x.size() =', context_x.size())
        test_y = test_y.view((test_data_num, 1, -1))  # (data_num, 1, y_dim)

        x, y = list_2_torch(self.dataset)
        # (data_num, x_dim), (data_num, y_dim)
        # NOTICE: The there is no batch_size or sequence length here
        #   .
        #   Image NP as gaussian process it should be (B=1, seq_len, x_dim)
        #   Image NP as a neural network it should be (B=?, seq_len=?, x_dim)
        #   Here I use it as a neural network
        #   The "batch_size"
        data_num = x.size()[0]
        # print(data_num)
        x = x.view((data_num, 1, -1))  # (data_num, 1, x_dim)
        y = y.view((data_num, 1, -1))  # (data_num, 1, y_dim)

        # forward the model to get loss
        # NOTICE: Batch size trick?  The context and target need to have the same batch_size
        #   but it is more reasonable to have "batchsize = 1"
        x = CUDA(x.view((1, -1, 6)))
        y = CUDA(y.view((1, -1, 5)))
        test_x = CUDA(test_x.view((1, -1, 6)))
        test_y = CUDA(test_y.view((1, -1, 5)))

        # TODO: config
        # NOTICE 1. Memory of function and data (Like GP)
        # mu, sigma, log_p, kl, loss = self.NP(x, y,
        #                                      test_x, test_y)

        # NOTICE 2. Memory of function (Like NN)
        mu, sigma, log_p, kl, loss = self.NP(test_x, test_y,
                                             test_x, test_y)

        # print('mu.size() =', mu.size())         # (1, test_num, y_dim)
        # print('sigma.size() =', sigma.size())   # (1, test_num, y_dim)

        mu_dist_v = mu.view(-1, 5)          # (test_num, y_dim)
        sigma_dist_v = sigma.view(-1, 5)    # (test_num, y_dim)
        cov = torch.diag_embed(sigma_dist_v)    # (test_num, y_dim, y_dim)
        mg = torch.distributions.MultivariateNormal(mu_dist_v, cov)
        ll = mg.log_prob(test_y.view((-1, 5)))
        # print('ll.size() =', ll.size())
        ll_mean = torch.mean(ll)
        ll_sum = torch.sum(ll)
        # print('ll_mean =', ll_mean)
        # print('ll_sum =', ll_sum)
        return loss, kl, ll_mean, ll_sum


def list_2_torch(data_list):
    '''
    :param data_list: [[index, array([0, 1, 2, 3], array([4]), array([5, 6, 7, 8])],
                       [index, array([0, 1, 2, 3], array([4]), array([5, 6, 7, 8])],
                       ...
                       [index, array([0, 1, 2, 3], array([4]), array([5, 6, 7, 8])]]
    :return:
    '''
    data_len = len(data_list)
    ctt_x_array = np.zeros((data_len, 6))
    ctt_y_array = np.zeros((data_len, 5))

    for i, data in enumerate(data_list):
        # print('i =', i)
        cluster_id, current_env, action, next_env = data
        ctt_x_array[i, 0:5] = current_env
        ctt_x_array[i, 5] = action
        ctt_y_array[i,:] = next_env
        # print('cluster_id =', cluster_id)
        # print(current_env)
        # print(action)
        # print(next_env)
        # print('ctt_x_array[i,:] =', ctt_x_array[i, :])
        # print('ctt_y_array[i, :] =', ctt_y_array[i, :])
        # break
    return torch.Tensor(ctt_x_array), torch.Tensor(ctt_y_array)




if __name__ == '__main__':
    print(1)
    # data = sin_wave_data()

    # data = np.load('data/data_gym_cartpole_preset.npy', allow_pickle=True)
    with open('data/data.pickle', 'rb') as f:
        data = pickle.load(f)
    # label = np.load('./data/data_gym_cartpole_transition_preset.npy', allow_pickle=True)

    # print('data =', data)
    # print('data[0] =', data[0])
    # print('data[0,1] =', data[0,1])
    # print('data.shape =', data.shape)   # (3079, 4)
    # print('type(data[0]) =', type(data[0])) #ndarray
    # print('len(data) =', len(data))

    # print('label =', label)

    # TODO: separate the data set into four parts
    #
    # with open('data.pickle', 'rb') as f:
    #     data = pickle.load(f)

    data_dict = {0: [], 1: [], 2: [], 3: []}
    for d in data:
        task_idx, obs, action, label = d
        data_dict[task_idx].append(d)

    for key in data_dict.keys():
        print("task %d has data %d" % (key, len(data_dict[key])))

    # print('(data_dict[0][0]) =', data_dict[0][0])   # list
    # type(data_dict[0][0]) = <class 'list'>

    # NOTICE: Here can separate each data_dict[i] into train and test sets
    # TODO: Add more randomness when selecting training and testing data
    data_train_dict = {}
    data_test_dict = {}
    train_precent = 0.5

    # # Not using rand perm while separating train and test
    # for i in [0, 1, 2, 3]:
    #     data_num = len(data_dict[i])
    #     train_data_num = int(train_precent*data_num)
    #     data_train_dict[i] = data_dict[i][:train_data_num]
    #     data_test_dict[i] = data_dict[i][train_data_num:]

    # rand perm
    for i in [0, 1, 2, 3]:
        data_num = len(data_dict[i])
        data_list = np.random.permutation(data_dict[i])
        # train_data_num = int(train_precent*data_num)
        train_data_num = int(200)
        data_train_dict[i] = data_list[:train_data_num]
        data_test_dict[i] = data_list[train_data_num:]




    # data0_ctt_x, data0_ctt_y = list_2_torch(data_dict[0])
    # print('data0_ctt_x.size() =', data0_ctt_x.size())
    # print('data0_ctt_y.size() =', data0_ctt_y.size())

    # dataset = []
    # for i in range(3):
    #     dataset += data_dict[i][::2]
    #
    # print('len(dataset) =', len(dataset))

    # NOTICE: Single data test
    #   ############################
    #   ############################
    #   ############################
    # NP_comp_0 = NP_component_test()
    # print('Start to fit NP_comp_0')
    # NP_comp_0.fit(data_dict[0], print_info=True)
    # # NP_comp_1.fit(data_dict[1])
    # # NP_comp_2.fit(data_dict[2])
    # # NP_comp_3.fit(data_dict[3])
    #
    # print("Test on 0")
    # loss_0 = NP_comp_0.likelihood_np(data_dict[0])
    # print('loss_0 =', loss_0)
    #
    # print("Test on 1")
    # loss_1 = NP_comp_0.likelihood_np(data_dict[1])
    # print('loss_1 =', loss_1)
    #
    # print("Test on 2")
    # loss_2 = NP_comp_0.likelihood_np(data_dict[2])
    # print('loss_2 =', loss_2)
    #
    # print("Test on 3")
    # loss_3 = NP_comp_0.likelihood_np(data_dict[3])
    # print('loss_3 =', loss_3)



    # NOTICE: Train 4 different NP
    #   ############################
    #   ############################
    #   ############################
    NP_comp_0 = NP_component_test()
    NP_comp_1 = NP_component_test()
    NP_comp_2 = NP_component_test()
    NP_comp_3 = NP_component_test()

    model_list = [NP_comp_0, NP_comp_1, NP_comp_2, NP_comp_3]

    for i in range(4):
        print('Fitting model ', i)
        model_list[i].fit(data_train_dict[i])

    # NOTICE: Done For each data, compare the "likelihood between from 4 NPs"
    # Test the whole episode

    for i in range(4):
        print()
        print('Testing data ', i)
        for m in range(4):
            # print('Testing data :', i)
            loss, kl, ll_mean, ll_sum = model_list[m].likelihood_np(data_test_dict[i])
            print('Test: The loss of model ', m, 'on data ', i,
                  'is loss= ',loss, 'kl=', kl, 'll_mean =', ll_mean, 'll_sum', ll_sum )

    for i in range(4):
        print()
        print('Training error ', i)
        for m in range(4):
            # print('Testing data :', i)
            tr_loss, tr_kl, ll_mean, ll_sum = model_list[m].likelihood_np(data_train_dict[i])
            print('Train: The tr_loss of model ', m, 'on data ', i,
                  'is loss= ',tr_loss, 'kl=', tr_kl, 'll_mean =', ll_mean, 'll_sum', ll_sum)



    # NOTICE: randomly get "episode data" from data
    #   and decide which cluster(env) does it belong to
    test_num = 100  # do 100 tests for each env
    test_observation_num = 150  # (number of data in a episode)

    # sample an episode from data_dict[i]
    for i in [0, 1, 2, 3]:
        # NOTICE:
        print()
        print("testing on", i, "data")
        all_data_num = len(data_dict[i])

        acc_loss_num = 0
        acc_ll_sum_num = 0

        for _ in range(test_num):
            loss_pred = [0, 0, 0, 0]
            ll_sum_pred = [0, 0, 0, 0]
            # sample an episode
            start_idx = np.random.randint(0, all_data_num - test_observation_num - 1)
            data_test_ep = data_dict[i][start_idx:start_idx + test_observation_num]

            # Compute the (loss, ll_max) of this data given each NP
            for m_i in [0, 1, 2, 3]:
                loss, kl, ll_mean, ll_sum = model_list[m_i].likelihood_np(data_test_ep)
                loss_pred[m_i] = loss
                ll_sum_pred[m_i] = ll_sum

            loss_pred_id = np.argmin(loss_pred)
            if loss_pred_id == i:
                acc_loss_num += 1

            ll_sum_pred_id = np.argmax(ll_sum_pred)
            if ll_sum_pred_id == i:
                acc_ll_sum_num += 1

        print('loss criteria pred acc =', acc_loss_num/test_num)
        print('ll_sum criteria pred acc =', acc_ll_sum_num/test_num)

