'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 01:02:16
@LastEditTime: 2020-03-24 10:50:21
@Description:
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class Model:
    
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, action):
        """
        Predict a batch of state and action pairs and return numpy array

        Parameters:
        ----------
            @param tensor or numpy - state : size should be (batch_size x state dim)
            @param tensor or numpy - action : size should be (batch_size x action dim)

        Return:
        ----------
            @param numpy array - state_next - size should be (batch_size x state dim)
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def fit(self, data, label):
        """
        Fit the model given data and label

        Parameters:
        ----------
            @param list of numpy array - data : each array size should be of (state dim + action dim)
            @param list of numpy array - label : each array size should be of (state dim)

        Return:
        ----------
            @param (int, int) - (training loss, test loss)
        """
        raise NotImplementedError("Must be implemented in subclass.")
