import tensorflow as tf
from sandbox.ignasi.maml.utils import normalize
import numpy as np
from tensorflow.python.platform import flags
import tensorflow.contrib.layers as layers

class DeterministicLSTMRegressor(object):
    def __init__(self, dim_input, dim_output, dim_hidden=(64, 64), steps=32, norm='None', dim_obs=None, max_path_length=1000,
                 **kwargs):

        # dims
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.dim_obs = dim_obs

        # initializers
        self.weight_initializer = tf.truncated_normal
        self.bias_initializer = tf.zeros
        self.mean_initializer = np.zeros
        self.std_initializer = np.ones

        # placeholders and variables
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.dim_input), name='test_inputs')
        self._update_params = self._update_params_data_dist(dim_input, dim_output)
        self._weights, self._inputs, self._output, self.output = None, None, None, None
        self.steps = steps
        self.max_path_length = max_path_length
        self.is_rnn = True
        self.is_oneshot = False
        self.multi_input = 0

        self.norm = norm
        self._construct_rnn()
        self._ch = [[tf.placeholder(tf.float32, shape=(None, dim)), tf.placeholder(tf.float32, shape=(None, dim))] for dim in self.dim_hidden]
        self._state = tuple([tf.nn.rnn_cell.LSTMStateTuple(c, h) for c,h in self._ch])
        # define the forward pass
        self._inputs = (self.inputs - self._x_mean_var) / self._x_std_var
        self._predict()
        self.weights = dict([('_'.join(v.name.split('/')).split(':')[0], v) for v in tf.trainable_variables()])
        self._weights = dict([(k, tf.placeholder(tf.float32, name=('ph_' + '_'.join(k.split('/')).split(':')[0]))) for k in self.weights.keys()])
        self._set_weights()
        self.state = None

    def _construct_rnn(self):
        cells = []
        for dim in self.dim_hidden:
            # NOTE: Default activation is tanh
            cells.append(tf.nn.rnn_cell.BasicLSTMCell(dim, state_is_tuple=True))
        self.cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        self.weights = {}
        self.weights['W0'] = tf.Variable(self.weight_initializer([self.dim_hidden[-1], self.dim_output], stddev=0.01), name='W0')
        self.weights['b0'] = tf.Variable(self.bias_initializer([self.dim_output]), name='b0')
        # _weights = dict([(v.name, v) for v in tf.trainable_variables()])

    def _predict(self):
        self._inputs = (self.inputs - self._x_mean_var) / self._x_std_var
        self._output, self.next_state = self.cell(self._inputs, self._state)
        self._output = tf.matmul(self._output, self.weights['W0']) + self.weights['b0']
        self.output = self._output * self._y_std_var + self._y_mean_var

    def forward_rnn(self, inp, *args, **kwargs):
        batch_size = tf.shape(inp)[0]
        state = self.cell.zero_state(batch_size, tf.float32)
        inp = tf.transpose(inp, (1, 0, 2))

        # Given inputs (time, batch, input_size) outputs a tuple
        #  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
        #  - states:  (time, batch, hidden_size)
        outputs = []
        for i in range(0, self.max_path_length, self.steps):
            rnn_outputs, state = tf.nn.dynamic_rnn(self.cell, inp[i:(i+self.steps)], initial_state=state, time_major=True)
            state = tuple([tf.contrib.rnn.LSTMStateTuple(tf.stop_gradient(c.c),
                                                         tf.stop_gradient(c.h))
                           for c in state])
            outputs.append(rnn_outputs)
        # TODO: Add linear layer to the outputs
        outputs = tf.concat(outputs, axis=0)
        final_projection = lambda x: tf.matmul(x, self.weights['W0']) + self.weights['b0']

        # apply projection to every timestep.
        outputs = tf.map_fn(final_projection, outputs)
        outputs = tf.transpose(outputs, (1, 0, 2))
        return outputs

    # run forward pass of NN using the given inputs
    def predict(self, *input_vals, first=True):
        sess = tf.get_default_session()
        inp, state = input_vals
        batch_size = input_vals[0].shape[0]
        if first:
            state = sum([[np.zeros((batch_size, dim))] * 2 for dim in self.dim_hidden], [])
        else:
            if type(state[0]) is tuple:
                state = sum([[s[0], s[1]] for s in state], [])
            else:
                state = sum([[s.c, s.h] for s in state], []) #TODO: Not sure
        ch = sum(self._ch, [])
        _state_fd = list(zip(ch, state))
        feed_dict = dict([(self.inputs, inp)] + _state_fd)
        return sess.run([self.output, self.next_state], feed_dict=feed_dict)

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
        x_mean = np.zeros(1, self.dim_input) if x_mean is None else x_mean
        x_std = np.ones(1, self.dim_input) if x_std is None else x_std
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
    def _update_params_data_dist(self, dim_input, dim_output):
        self._x_mean_var = tf.Variable(
            self.mean_initializer((1, dim_input)), dtype=np.float32,
            trainable=False,
            name="x_mean",
        )
        self._x_std_var = tf.Variable(
            self.std_initializer((1, dim_input)), dtype=np.float32,
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
                                         shape=(1, dim_input),
                                         name="x_mean_ph",
                                         )
        self._x_std_ph = tf.placeholder(tf.float32,
                                        shape=(1, dim_input),
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
                                                      + new_value * self._nb_ph)/tf.add(self._nt,self._nb_ph)
        _update_params = [
            tf.assign(self._x_mean_var, running_mean(self._x_mean_var, self._x_mean_ph)),
            tf.assign(self._x_std_var, running_mean(self._x_std_var, self._x_std_ph)),
            tf.assign(self._y_mean_var, running_mean(self._y_mean_var, self._y_mean_ph)),
            tf.assign(self._y_std_var, running_mean(self._y_std_var, self._y_std_ph)),
            tf.assign(self._nt, tf.add(self._nt, self._nb_ph)),
        ]
        return _update_params



