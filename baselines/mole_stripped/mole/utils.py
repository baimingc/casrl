""" Utility functions. """

from tf.envs.base import TfEnv

import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
import glob, signal, shutil, sys
import os.path as osp

from rllab.envs.normalized_env import normalize as normalize_env

# from mole.envs.half_cheetah_env import HalfCheetahEnv
from mole.envs.half_cheetah_actions_env import HalfCheetahActionsEnv
# from mole.envs.half_cheetah_hfield_env import HalfCheetahEnv as HalfCheetahHFieldEnv
# from mole.envs.half_cheetah_blocks_env import HalfCheetahBlocksEnv
# from mole.envs.ant_env import AntEnv
# from mole.envs.crawler_env import CrawlerEnv
# from mole.envs.arm_7dof_env import Arm7DofEnv

def create_env(env_name, task):

    if env_name == 'HalfCheetahActionsEnv':
        env = TfEnv(normalize_env(HalfCheetahActionsEnv(task=task, reset_every_episode=True, reward=True)))
        agent_type='cheetah_ignore3'
    else:
        raise (NotImplementedError, 'Environment not implemented yet!')
    return env, agent_type

####################

def flatten(l):
    return [item for sublist in l for item in sublist]

#for replacing values from config file with commandline args
def replace_in_dict(config, extra_config):
    for extra_k, extra_v in extra_config.items():
        found = False
        for k, v in config.items():
            if type(v) is dict:
                if extra_k in v.keys():
                    found = True
                    config[k][extra_k] = extra_v
            else:
                if extra_k == k:
                    found = True
                    config[k] = extra_v
        if not found:
            print("\n\nKEY: ", extra_k)
        assert found, 'Key not found!'
    return config

# Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images


# Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed


def normalize(inp, activation, reuse, scope, norm):
    if norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope, is_training=FLAGS.train)
    elif norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'None':
        return activation(inp)


# Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))


def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size

class Exit:
    def __init__(self):
        self.path = ''
        self.original_sigint = signal.getsignal(signal.SIGINT)

    def num_meta_files(self):
        if osp.isdir(self.path):
            files = glob.glob(osp.join(self.path, '*.meta'))
            return len(files)
        else:
            return -1

    def exit_gracefully(self, signum, frame):
        # restore the original signal handler as otherwise evil things will happen
        # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
        signal.signal(signal.SIGINT, self.original_sigint)

        n = self.num_meta_files()
        try:
            if input("\nReally quit? (y/n)> ").lower().startswith('y'):
                if n == 0:
                    print('No model saved, removing the folder')
                    shutil.rmtree(self.path)
                sys.exit(1)

        except:
            if n == 0:
                print('No model saved, removing the folder')
                shutil.rmtree(self.path)
            sys.exit(1)

        # restore the exit gracefully handler here
        signal.signal(signal.SIGINT, self.exit_gracefully)


class PlotRegressor(object):
    def __init__(self, env_name):
        self.env_name = env_name
        self.joints = self.get_important_joints()

    def get_important_joints(self):
        if self.env_name == 'CrawlerEnv':
            return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        elif self.env_name in ['HalfCheetahEnv', 'HalfCheetahBlocksEnv']:
            return np.array([15, 16, 17, 18, 19, 20])
        else:
            raise NotImplementedError

    def plot(self, obs, pred_obs, y_std_var):
        
        plot_error=False
        sess = tf.get_default_session()

        if(plot_error):

            #calculate errors
            std_error = (np.array(obs) - np.array(pred_obs)) / sess.run(y_std_var)
            print('Average loss', 0.5 * np.mean(np.square(std_error[1:])))
            print('Max loss: ', np.max(np.abs(std_error[1:])))
            x = np.tile(np.arange(std_error.shape[0]), (len(self.joints), 1)).T
            
            #plot all errors in 1 plot
            plt.plot(x, std_error[:, self.joints])
            plt.title('Normalized Error')

        else:

            #init vars
            f, ax = plt.subplots(len(self.joints), 1)
            colors=['black', 'blue', 'red', 'teal', 'green', 'yellowgreen', 'maroon']
            true = np.array(obs).T
            pred = np.array(pred_obs).T
            color_counter=0
            counter=0

            #plot each joint of interest (true vs predicted)
            for i in range(true.shape[0]):
                if(i in self.joints):
                    ax[counter].plot(true[i], color=colors[color_counter])
                    ax[counter].plot(pred[i], '--', color=colors[color_counter])
                    ax[counter].set_title(i)
                    color_counter+=1
                    counter+=1
                    if(color_counter>=len(colors)):
                        color_counter=0

            plt.title('True (solid) vs Predicted (dashed)')

        plt.show()
        plt.close()