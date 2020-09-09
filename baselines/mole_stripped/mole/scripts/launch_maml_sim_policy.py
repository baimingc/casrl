import argparse
import os.path as osp
import yaml
import glob
import numpy as np
import time

from rllab.envs.normalized_env import normalize

from tf.misc import tensor_utils
from tf.envs.base import TfEnv

from mole.algos.metatest_maml import MetaTest
from mole.maml import MAML
from mole.maml_continual import MAML_continual
from mole.maml_continual_truncBackprop import MAML_continual_truncBackprop
from mole.policies.naive_mpc_controller import NaiveMPCController
from mole.policies.uniform_control_policy import UniformControlPolicy
from mole.regressors.deterministic_mlp_regressor import DeterministicMLPRegressor
from mole.utils import replace_in_dict, create_env

import tensorflow as tf

def main(args):

    #seed
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    #vars
    testing_a_continuallyTrained_model = args.cont_model
    dontResetBetwRollouts = args.dontResetBetwRollouts
    tempLR = args.tempLR
    temp = args.temp

    #configs
    test_config = {'animated': args.render, 'max_path_length': args.length,
                    'speedup': args.speedup, 
                    'num_rollouts': args.num_rollouts}
    config_overwrite = {'max_path_length': args.length, 
                        'task': args.task} 

    #read in config vars from the training
    config = yaml.load(open(osp.join(args.filename, 'config.yaml')))
    model_file = osp.join(args.filename, 'model'+ str(args.model_num))
    print("\n\nFILENAME: ", model_file)

    #overwrite saved params from config with the ones specified in this script
    config = replace_in_dict(config, config_overwrite)

    #print info
    print("\n\n*************** TASK: ", args.task)
    print("*************** tempLR: ", tempLR)
    print("*************** temp: ", temp)
    print("*************** dontResetBetwRollouts: ", dontResetBetwRollouts)

    if testing_a_continuallyTrained_model: print("\n\n**** LOADING A CONTINUAL MODEL")
    else: print("\n\n**** LOADING A NON-CONTINUAL MODEL")

    if args.thetaResetChoice==5: print("**** testing in continual way...\n\n")
    else: print("**** testing in non-continual way, option: ", str(args.thetaResetChoice) , "\n\n")

    #some params
    ignore_absolute_xy = config['training']['ignore_absolute_xy']
    config['training']['update_lr'] = tempLR
    test_config['update_lr'] = tempLR

    ##########################################
    ## CREATE env, regressor, policy, maml model
    ##########################################

    # create env
    env, _ = create_env(config['env']['name'], args.task)

    # init vars
    obs_dim = env.observation_space.flat_dim
    act_dim = env.action_dim
    dim_output = obs_dim
    dim_input = obs_dim + act_dim
    dim_input_regressor= obs_dim + act_dim

    # hacky... ignore some dims
    # TODO set agent_type somehow w env name (check w train script for consistency)
    agent_type = 'todo'
    if(ignore_absolute_xy):
        if(agent_type=='cheetah_hfield'):
            dim_input_regressor -= 1
        elif(agent_type=='cheetah_ignore3'):
            dim_input_regressor -= 3
        elif(agent_type=='roach_ignore4'):
            dim_input_regressor -= 4
        else:
            dim_input_regressor -= 2

    #create regressor
    regressor = DeterministicMLPRegressor(dim_input_regressor, dim_input, dim_output, 
                                        ignore_absolute_xy=ignore_absolute_xy, agent_type=agent_type, **config['model'])

    #create policy
    policy = NaiveMPCController(env, regressor, **config['policy'])

    #set ubs
    if testing_a_continuallyTrained_model:
        k = 10
        chunk_size_into_model = 2*k
    else:
        #NOTE: this ubs represents 2k
        # TODO read the code to understand this again + clean it up here... TODO assign this from config
        config['training']['update_batch_size'] = 20 
        chunk_size_into_model = config['training']['update_batch_size']

        if args.thetaResetChoice==5: 
            k = int(chunk_size_into_model/2.0)
        else: 
            k = chunk_size_into_model
        print("thinks ubs is: ", chunk_size_into_model)

    # create MAML model
    if testing_a_continuallyTrained_model:
        print("ERROR... this codebase doesn't currently have the ability to have trained something in a continual way, so this is not a valid option...")
        raise NotImplementedError
        # TRUNCATED = False
        # BACKWARD_LENGTH = 10
        # temp = config['training']['temp']
        # validation_set_size = 16

        # if(TRUNCATED):
        #     model = MAML_continual_truncBackprop(regressor, k, validation_set_size, ignore_absolute_xy,
        #                         agent_type, dim_input_regressor, dim_input, dim_output, config['model']['dim_bias'], 
        #                         BACKWARD_LENGTH=BACKWARD_LENGTH,
        #                         config=config['training'])
        # else:
        #     model = MAML_continual(regressor, k, validation_set_size, ignore_absolute_xy,
        #                         agent_type, dim_input_regressor, dim_input, dim_output, 
        #                         config['model']['dim_bias'], config=config['training'])
    else:
        model = MAML(regressor, dim_input, dim_output, num_extra=0, config=config['training'])

    # construct the tf graph
    print("\n\n...START constructing tensorflow graph")
    construction_start_time = time.time()
    model.construct_model(input_tensors=None)
    elapsed_time = time.time() - construction_start_time
    print("...DONE constructing tensorflow graph... time taken: ", elapsed_time, "\n\n")

    ############################
    ## INIT tf, weights
    ############################

    #start tf session
    sess = tf.InteractiveSession()

    #tf init
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    try:
        print("Restoring model weights from " + args.filename)
        saver.restore(sess, model_file)
    except:
        raise NameError

    ##########################################
    ## TEST
    ##########################################
    test_config['render'] = None
    metatest = MetaTest(chunk_size_into_model, k, model, args.thetaResetChoice, tempLR, temp, dontResetBetwRollouts, 
                    args.do_live_plot, args.filename, testing_a_continuallyTrained_model=testing_a_continuallyTrained_model,
                    config=test_config, tasktype=args.task, num_updates=config['training']['num_updates'])

    metatest.metatest(env, policy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('--model_num', type=int, default=15) 
    parser.add_argument('-n', '--thetaResetChoice', type=int, default=5) #2 always reset to theta*, 5 ours, 3 always take gradient step
    parser.add_argument('-t', '--task', type=str, default='cripple_set3')
    parser.add_argument('-l', '--length', type=int, default=2000) 
    parser.add_argument('-tempLR', '--tempLR', type=float, default=0.01) 
    parser.add_argument('-temp', '--temp', type=float, default=0.02)  #0.02, higher means more peaky
    parser.add_argument('-speedup', '--speedup', type=int, default=1)
    parser.add_argument('--num_rollouts', type=int, default=1)
    parser.add_argument('-seed', '--seed', type=int, default=0)
    parser.add_argument('-render', '--render', action='store_true')
    parser.add_argument('-do_live_plot', '--do_live_plot', action='store_true')
    parser.add_argument('-testing_a_continuallyTrained_model', '--cont_model', action='store_true')
    parser.add_argument('-dontResetBetwRollouts', '--dontResetBetwRollouts', action='store_true')
    args = parser.parse_args()
    main(args)
