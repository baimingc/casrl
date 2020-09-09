import os
import numpy as np
from loguru import logger
import pickle
import yaml
print('yaml.__version__ =', yaml.__version__)
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from DPNPMM import DPNPMM

from matplotlib import pyplot as plt

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

config = load_config('config_test.yml')
# print('config =', config)
nn_config = config['NN_config']
mpc_config = config['mpc_config']
gym_config = config['gym_config']
dp_config = config["DP_config"]
np_config = config["NP_config"]


# with open('data/data.pickle', 'rb') as f:
#     data = pickle.load(f)

# Still use cartpole data
data = np.load('data/data_gym_cartpole_preset.npy', allow_pickle=True)

print('type(data) =', type(data))
data_dict = {0:[],1:[],2:[],3:[]}
for d in data:
    task_idx, obs, action, label = d
    data_dict[task_idx].append(d)

for key in data_dict.keys():
    print("task %d has data %d"%(key, len(data_dict[key])))

# TODO List:
#   (1) Check the data, output the data labels
#   (2) Test if fast train or meta-train can approx prior in a stable way
#   (3) Automatic BNP cluster of streaming data
# id_list = []
# for data_point in data:
#     print('Id = ',data_point[0])
#     id_list.append(data_point[0])
#
# plt.plot(id_list)
# plt.show()

# NOTICE: for cartpole seq_len = 20
#  seq_len = 20
#  ################################
seq_len = 20

# NOTICE: Create data list
#   note, the first 80 datapoints shoule be different clusters
#   so that we can investigate whether DPNP can cluster data properly

total_data_num = 800
seq_num = total_data_num/seq_len
data_for_cluster = []

data = data.tolist()
last_id = -1
for i, point in enumerate(data):
    if (point[0] != last_id) and (i + 20 < len(data)):
        data_for_cluster += data[i: i+20]
        last_id = point[0]

# for point in data_for_cluster:
#     print(point[0])
#   NOTICE: Got data for cluster, and the data is relative sequential
#    data_for_cluster

DPNPmodel = DPNPMM(dp_config, np_config)

seq_count = 0
true_id_list = []
pred_id_list_rec = []
true_id_list_rec = []
for i, data in enumerate(data):
    true_id_list.append(data[0])
    DPNPmodel.add_data_point(data)
    if DPNPmodel.stm_is_full:
        print()
        print('**********************SVI******************')
        print('true_id_list is =', true_id_list)
        true_id_list_rec = true_id_list_rec + true_id_list

        seq_count += 1
        time_use, task_idx_pred = DPNPmodel.fit()
        print('time_use =', time_use)
        print('task_idx_pred =', task_idx_pred)

        true_id_list = []

        pred_id_list_rec = pred_id_list_rec + [task_idx_pred]*20
        if seq_count == 30:
            break

figure = plt.figure(1)
plt.plot(true_id_list_rec, label='True id')
plt.plot(pred_id_list_rec, label='Pred id')
plt.legend()
plt.show()
