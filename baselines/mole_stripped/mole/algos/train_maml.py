import numpy as np
import tensorflow as tf
import random
import pickle
import csv
from tensorflow.python.platform import flags
import time
import os.path as osp
import os
import yaml
import rllab.misc.logger as logger
from rllab import config


def train(model, saver, sess, exp_string, data_generator, config, resume_itr=1, num_extra=0, variable_num_sgd_steps=False, which_timeseries='household'):
  
  num_updates=1

  #get certain sections of vars from config file
  log_config = config['logging']
  train_config = config['training']

  #init vars
  t0 = time.time()
  path = logger.get_snapshot_dir() #osp.join(log_config['log_dir'], exp_string)
  # os.makedirs(path)
  if log_config['log']:
      # logger.add_tabular_output(osp.join(log_config['log_dir'], exp_string, 'progress.csv'))
      train_writer = tf.summary.FileWriter(path, sess.graph)
  print('Done initializing, starting training.')

  #save the config params to a file in the save directory
  yaml.dump(config, open(osp.join(path, 'config.yaml'), 'w'))

  #init empty lists
  prelosses, postlosses, prelosses_val, postlosses_val = [], [], [], []
  multitask_weights, reg_weights = [], []

  #######################################
  ## loop for meta-training iterations
  #######################################
  agg_itr = 0
  firstTime=True
  gradient_step = 0

  previous_epoch = 0
  inputa_val=None
  all_prelosses=[]
  all_postlosses=[]
  all_prelosses_val=[]
  all_postlosses_val=[]
  all_gradsteps=[]

  while(agg_itr < train_config['metatrain_itr']):

    #############################
    ## retrieve samples
    #############################

    # as long as the selected data_generator has a generate func,
      # retrieve samples of inputs and outputs
      # [meta_bs, 2*update_bs, dim]
    if data_generator._epochs >= data_generator.max_epochs:
        firstTime=False
        agg_itr += 1
        name = 'model' + str(agg_itr)
        print('Saving model')
        saver.save(sess,  osp.join(path, name))
        previous_epoch=0
    if 'generate' in dir(data_generator):
        batch_x, batch_y = data_generator.generate(firstTime, num_updates)

    #use the 1st half as training data for inner-loop gradient
    inputa = batch_x[:, :train_config['update_batch_size']+num_extra, :]
    labela = batch_y[:, :train_config['update_batch_size']+num_extra, :]
    #use the 2nd half as test data for outer-loop gradient
    inputb = batch_x[:, train_config['update_batch_size']+num_extra:, :]
    labelb = batch_y[:, train_config['update_batch_size']+num_extra:, :]

    #############################
    ## run meta-training iteration
    #############################

    if(variable_num_sgd_steps):
      num_sgd_steps= np.random.randint(1,6) #[low, high) (1,6) and (1,11)
      num_updates= 1 ##################np.random.randint(1,5) 
    else:
      num_sgd_steps= 1
      num_updates= 1

    #define which tensors to populate/execute during the sess.run call
    if gradient_step < train_config['pretrain_itr']:
        input_tensors = [model.pretrain_op]
    else:
        input_tensors = [model.metatrain_op]
    if gradient_step % log_config['print_itr'] == 0 or gradient_step % log_config['summary_itr'] == 0:
        input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2])

    #make the sess.run call to perform one metatraining iteration
    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb, model.num_sgd_steps: num_sgd_steps, model.num_updates: num_updates}
    result = sess.run(input_tensors, feed_dict)

    #############################
    ## calculate validation loss
    #############################

    #make the sess.run call to check validation loss of the new parameters when they adapt
    # feed_dict = {model.inputa: inputa_val, model.inputb: inputb_val, model.labela: labela_val, model.labelb: labelb_val, model.num_sgd_steps: num_sgd_steps, model.num_updates: num_updates}
    # result_val = sess.run([model.total_loss1, model.total_losses2], feed_dict)

    #############################
    ## logging and saving
    #############################

    if gradient_step % log_config['summary_itr'] == 0:
        prelosses.append(result[2])
        # prelosses_val.append(result_val[0])
        # postlosses_val.append(result_val[1])
        if log_config['log']:
            train_writer.add_summary(result[1], gradient_step)
        postlosses.append(result[3][-1])

    if gradient_step % log_config['print_itr'] == 0:
        if gradient_step < train_config['pretrain_itr']:
            print_str = 'Pretrain Iteration ' + str(gradient_step)
        else:
            print_str = 'Iteration ' + str(gradient_step - train_config['pretrain_itr'])
        print_str += '   | Mean pre-losses: ' + str(np.mean(prelosses)) + '   | Mean post-losses: ' + str(np.mean(postlosses))
        # print_str += '   | Val pre-losses: ' + str(np.mean(prelosses_val)) + '   | Val post-losses: ' + str(np.mean(postlosses_val))
        print_str += '    | Time spent:   {0:.2f}'.format(time.time() - t0)
        print(print_str)
        t0 = time.time()

        #save for plotting
        all_prelosses.append(np.mean(prelosses))
        all_postlosses.append(np.mean(postlosses))
        # all_prelosses_val.append(np.mean(prelosses_val))
        # all_postlosses_val.append(np.mean(postlosses_val))
        all_gradsteps.append(gradient_step)
        prelosses, postlosses, prelosses_val, postlosses_val = [], [], [], []

    if(gradient_step%2000==0):
      np.save(osp.join(path, 'saved_prelosses.npy'), all_prelosses)
      np.save(osp.join(path, 'saved_postlosses.npy'), all_postlosses)
      # np.save(osp.join(path, 'saved_prelosses_val.npy'), all_prelosses_val)
      # np.save(osp.join(path, 'saved_postlosses_val.npy'), all_postlosses_val)
      np.save(osp.join(path, 'saved_all_gradsteps.npy'), all_gradsteps)

    gradient_step += 1

    if(gradient_step%50==0):
      print("   gradient step: ", gradient_step)

