import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
from mole.utils import normalize

class DeterministicMLPRegressor(object):
    def __init__(self, dim_input, dim_input_full, dim_output, dim_hidden=(64, 64), dim_conv1d=(8, 8, 8), norm='None', dim_obs=None,
                 dim_bias=0, multi_input=0, ignore_absolute_xy=False, agent_type='cheetah'):

        # dims
        self.multi_input = multi_input
        if self.multi_input:
            self.dim_input = dim_input * self.multi_input
        else:
            self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.dim_conv1d = dim_conv1d
        self.dim_obs = dim_obs
        self.dim_bias = dim_bias

        self.dim_input_full = dim_input_full

        # initializers
        self.weight_initializer = tf.truncated_normal
        self.bias_initializer = tf.zeros
        self.mean_initializer = np.zeros
        self.std_initializer = np.ones
        self.is_rnn = False
        self.is_oneshot = False

        self.ignore_absolute_xy=ignore_absolute_xy
        self.agent_type=agent_type

        # placeholders and variables
        self.inputs = tf.placeholder(tf.float32, shape=(None, None), name='test_inputs')
        self._update_params = self._update_params_data_dist(dim_input_full, dim_output)
        self.weights, self._inputs, self._output, self.output = None, None, None, None
        self._weights = None

        self.norm = norm

    def construct_fc_weights(self, meta_loss=False):
        self.weights = self._construct_fc_weights(meta_loss=meta_loss)
        # define the forward pass
        self._inputs = (self.inputs - self._x_mean_var) / self._x_std_var
        self._output = self.forward_fc(self._inputs, self.weights)
        self.output = self._output * self._y_std_var + self._y_mean_var
        self._weights = dict([(k, tf.placeholder(tf.float32, name=('ph_'+k))) for k in self.weights.keys()])
        self._set_weights()

    def _construct_fc_weights(self, meta_loss):
        weights = dict()
        if self.dim_bias > 0:
            weights['bias'] = tf.Variable(self.bias_initializer([self.dim_bias]))

        # the 1st hidden layer
        weights['W0'] = tf.Variable(self.weight_initializer([self.dim_input + self.dim_bias, self.dim_hidden[0]], stddev=0.01))
        weights['b0'] = tf.Variable(self.bias_initializer([self.dim_hidden[0]]))

        # intermediate hidden layers
        for i in range(1, len(self.dim_hidden)):
            weights['W' + str(i)] = tf.Variable(
                self.weight_initializer([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i)] = tf.Variable(self.bias_initializer([self.dim_hidden[i]]))

        weights['W' + str(len(self.dim_hidden))] = tf.Variable(
            self.weight_initializer([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b' + str(len(self.dim_hidden))] = tf.Variable(self.bias_initializer([self.dim_output]))
        if meta_loss:
            for i in range(len(self.dim_conv1d)):
                weights['W_1d_conv' + str(i)] = tf.Variable(
                    self.weight_initializer([self.dim_conv1d[i], self.dim_output, self.dim_output], stddev=0.01))
                weights['b_1d_conv' + str(i)] = tf.Variable(self.bias_initializer([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False, meta_loss=False, dontTruncate=False):

        #dont pass x,y absolute position through the neural net
        if(self.ignore_absolute_xy):
            if(self.agent_type=='cheetah'):
                inp= tf.concat([inp[:,:-3], tf.expand_dims(inp[:,-1], axis=1)], axis=1)
            elif(self.agent_type=='cheetah_hfield'):
                inp= tf.concat([inp[:,:-3], tf.expand_dims(inp[:,-2], axis=1), tf.expand_dims(inp[:,-1], axis=1)], axis=1)
            elif(self.agent_type=='arm'):
                pass
            elif(self.agent_type=='ant'):
                inp= tf.concat([inp[:,:-3], tf.expand_dims(inp[:,-1], axis=1)], axis=1)
            elif(self.agent_type=='cheetah_ignore3'):
                inp= tf.concat([inp[:,1:-3], tf.expand_dims(inp[:,-1], axis=1)], axis=1) 
                #ignore 0, -3, -2
                    #thought it would by rootx, xcom, ycom
                    #but it was really rootz, xcom, ycom
            elif(self.agent_type=='roach_ignore4'):
                inp= tf.concat([tf.expand_dims(inp[:,0], axis=1), inp[:,3:-3], tf.expand_dims(inp[:,-1], axis=1)], axis=1) #1,2,-3,-2

            else:
                print("\n\nNOT IMPLEMENTED ignoring xy in regressor")
                import IPython
                IPython.embed()

        # pass through initial bias
        if self.dim_bias > 0:
            n = tf.shape(inp)[0]
            bias = tf.tile(tf.expand_dims(weights['bias'], 0), (tf.to_int32(n), 1))
            hidden = tf.concat([inp, bias], axis=-1)
        else:
            hidden = inp

        # pass through intermediate hidden layers    
        for i in range(0, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['W' + str(i)]) + weights['b' + str(i)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i), norm=self.norm)

        # pass through output layer
        out = tf.matmul(hidden, weights['W' + str(len(self.dim_hidden))]) + weights[
            'b' + str(len(self.dim_hidden))]
        if meta_loss:
            out = tf.reshape(out, [-1, out.get_shape().dims[-2].value, self.dim_output])
            for i in range(len(self.dim_conv1d) - 1):
                out = normalize(tf.nn.conv1d(out,  weights['W_1d_conv' + str(i)], 1,
                                'SAME', data_format='NHWC', name='conv1d'+str(i), use_cudnn_on_gpu=True) +
                                weights['b_1d_conv'+str(i)], activation=tf.nn.relu,
                                reuse=reuse, scope='conv1d'+str(i), norm=self.norm)
            out = tf.nn.conv1d(out,  weights['W_1d_conv' + str(len(self.dim_conv1d)-1)], 1,
                               'SAME', data_format='NHWC', name='conv1d'+str(len(self.dim_conv1d)-1), use_cudnn_on_gpu=True) +\
                  weights['b_1d_conv'+str(len(self.dim_conv1d)-1)]
        return out

    # run forward pass of NN using the given inputs
    def predict(self, *input_vals):
        sess = tf.get_default_session()
        feed_dict = dict(list(zip([self.inputs], input_vals)))
        return sess.run(self.output, feed_dict=feed_dict)

    # evaluate and return the NN's weights and mean/std vars
    def get_params(self):
        sess = tf.get_default_session()
        return sess.run(self.weights)

    def _set_weights(self):
        self.set_weights = [tf.assign(self.weights[k], self._weights[k]) for k in self.weights.keys()]

    # update the NN's weights and mean/std vars
    def set_params(self, params):
        sess = tf.get_default_session()
        weights = params
        feed_dict = dict([(self._weights[k], weights[k]) for k in weights.keys()])
        _ = sess.run(self.set_weights, feed_dict=feed_dict)

    # helper function to update the NN's mean/std vars
    def update_params_data_dist(self, x_mean=None, x_std=None, y_mean=None, y_std=None, nb=0):
        x_mean = np.zeros(1, self.dim_input_full) if x_mean is None else x_mean
        x_std = np.ones(1, self.dim_input_full) if x_std is None else x_std
        y_mean = np.zeros(1, self.dim_output) if y_mean is None else y_mean
        y_std = np.ones(1, self.dim_output) if y_std is None else y_std
        feed_dict = {self._x_mean_ph: x_mean,
                     self._x_std_ph: x_std,
                     self._y_mean_ph: y_mean,
                     self._y_std_ph: y_std,
                     self._nb_ph: nb}
        sess = tf.get_default_session()
        _ = sess.run(self._update_params, feed_dict=feed_dict)

    # initialize the NN's mean/std vars
    # and provide assign commands to be able to update them when desired
    def _update_params_data_dist(self, dim_input_full, dim_output):
        if self.multi_input:
         multi = self.multi_input
        else:
          multi = 1
        self._x_mean_var = tf.Variable(
            self.mean_initializer((1, multi * dim_input_full)), dtype=np.float32,
            trainable=False,
            name="x_mean",
        )
        self._x_std_var = tf.Variable(
            self.std_initializer((1, multi * dim_input_full)), dtype=np.float32,
            trainable=False,
            name="x_std",
        )
        self._y_mean_var = tf.Variable(
            self.mean_initializer((1, dim_output)), dtype=np.float32,
            trainable=False,
            name="y_mean",
        )
        self._y_std_var = tf.Variable(
            self.std_initializer((1, dim_output)), dtype=np.float32,
            trainable=False,
            name="y_std",
        )
        self._nt = tf.Variable(0., dtype=np.float32,
                               trainable=False,
                               name='nt')

        self._x_mean_ph = tf.placeholder(tf.float32,
                                         shape=(1, dim_input_full),
                                         name="x_mean_ph",
                                         )
        self._x_std_ph = tf.placeholder(tf.float32,
                                        shape=(1, dim_input_full),
                                        name="x_std_ph",
                                        )
        self._y_mean_ph = tf.placeholder(tf.float32,
                                         shape=(1, dim_output),
                                         name="y_mean_ph",
                                         )
        self._y_std_ph = tf.placeholder(tf.float32,
                                        shape=(1, dim_output),
                                        name="y_std_ph",
                                        )
        self._nb_ph = tf.placeholder(tf.float32, shape=(), name="nb")
        running_mean = lambda curr_value, new_value: (curr_value * self._nt
                                                      + new_value * self._nb_ph)/tf.add(self._nt, self._nb_ph)
        if self.multi_input:
            _update_params = [
              tf.assign(self._x_mean_var, running_mean(self._x_mean_var, tf.tile(self._x_mean_ph, (1, self.multi_input)))),
              tf.assign(self._x_std_var, running_mean(self._x_std_var, tf.tile(self._x_std_ph, (1, self.multi_input)))),
              tf.assign(self._y_mean_var, running_mean(self._y_mean_var, self._y_mean_ph)),
              tf.assign(self._y_std_var, running_mean(self._y_std_var, self._y_std_ph)),
              tf.assign(self._nt, tf.add(self._nt, self._nb_ph)),]
        else:
            _update_params = [
                tf.assign(self._x_mean_var, running_mean(self._x_mean_var, self._x_mean_ph)),
                tf.assign(self._x_std_var, running_mean(self._x_std_var, self._x_std_ph)),
                tf.assign(self._y_mean_var, running_mean(self._y_mean_var, self._y_mean_ph)),
                tf.assign(self._y_std_var, running_mean(self._y_std_var, self._y_std_ph)),
                tf.assign(self._nt, tf.add(self._nt, self._nb_ph)),]
        return _update_params


