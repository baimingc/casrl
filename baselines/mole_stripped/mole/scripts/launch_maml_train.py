import tensorflow as tf
import random
import argparse
import os.path as osp
import yaml
from datetime import datetime
import signal
import os
import time

from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator

from mole.maml import MAML
from mole.algos.train_maml import train
from mole.policies.naive_mpc_controller import NaiveMPCController as Policy
from mole.regressors.deterministic_mlp_regressor import DeterministicMLPRegressor
from mole.sampler.sampler import BatchPolopt
from mole.utils import Exit, replace_in_dict, create_env


def run(d):

    # create env
    config = d['config']
    exp_name = d['exp_name']

    env, agent_type = create_env(config['env']['name'], config['env']['task'])

    #variables from config
    ignore_absolute_xy= config['training']['ignore_absolute_xy'] 
    variable_num_sgd_steps= False
    add_elems_for_sequential_sgd_steps = 0

    # set vars
    obs_dim = env.observation_space.flat_dim
    act_dim = env.action_dim
    dim_output = obs_dim
    dim_input = obs_dim + act_dim

    ###########################################################
    ## CREATE regressor, policy, data generator, maml model
    ###########################################################

    # create regressor (NN dynamics model)
    dim_input_regressor= obs_dim + act_dim
    if(ignore_absolute_xy):
        if(agent_type=='cheetah_hfield'):
            dim_input_regressor -= 1
        elif(agent_type=='cheetah_ignore3'):
            dim_input_regressor -= 3
        elif(agent_type=='roach_ignore4'):
            dim_input_regressor -= 4
        else:
            dim_input_regressor -= 2

    print("\n\n************")
    print("full input dim: ", dim_input)
    print("regressor input dim: ", dim_input_regressor)
    print("regressor action dim: ", act_dim)
    print("regressor output dim: ", dim_output)
    regressor = DeterministicMLPRegressor(dim_input_regressor, dim_input,
                                          dim_output, **config['model'], dim_obs=obs_dim,
                                          multi_input=config['sampler']['multi_input'],
                                          ignore_absolute_xy=ignore_absolute_xy, agent_type=agent_type)
    # create policy (MPC controller)
    policy = Policy(env, regressor, **config['policy'])

    # create MAML model
    model = MAML(regressor, dim_input, dim_output, num_extra=add_elems_for_sequential_sgd_steps, config=config['training'])

    # construct the tf graph
    print("\n\n...START constructing tensorflow graph")
    construction_start_time = time.time()
    model.construct_model(input_tensors=None, prefix='metatrain_')
    elapsed_time = time.time() - construction_start_time
    print("...DONE constructing tensorflow graph... time taken: ", elapsed_time, "\n\n")
    model.summ_op = tf.summary.merge_all()

    # create data generator for collecting rollouts
    data_generator = BatchPolopt(env, policy, regressor, model, scope='algo',
                             meta_bs=config['training']['meta_batch_size'],
                             update_bs=config['training']['update_batch_size'] * 2 + add_elems_for_sequential_sgd_steps,
                             maml=True, num_extra=add_elems_for_sequential_sgd_steps, 
                             not_doing_mbrl=False, which_timeseries=None, time_series_data=None,
                             **config['sampler'])

    # GPU config proto
    gpu_frac = 0.4
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    gpu_config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True,
                            inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    saver = tf.train.Saver(max_to_keep=20)
    sess = tf.InteractiveSession(config=gpu_config)

    # initialize tensorflow vars
    init_list = [v for v in tf.all_variables() if v not in tf.get_collection("dontInit")]
    tf.variables_initializer(init_list).run()
    tf.train.start_queue_runners()

    # train the regressor
    train(model, saver, sess, exp_name, data_generator, config, 
        resume_itr=0, num_extra=add_elems_for_sequential_sgd_steps, 
        variable_num_sgd_steps=variable_num_sgd_steps, which_timeseries=None)


def main(config_path, use_gpu):

    #################################
    ## INIT config and vars
    #################################

    #read in config vars
    config = yaml.load(open(config_path))

    vg = VariantGenerator()
    vg.add('config', [config])


    ### NOTE: you can add params here, to overwrite the values in config, like this:
    # vg.add('batch_size', [2000])
    # vg.add('update_batch_size', [8])


    for v in vg.variants():

        # remove some entries, because this is what will be used for the filename for saving
        v_usedfornaming = v.copy()
        del v_usedfornaming['config']
        del v_usedfornaming['_hidden_keys']

        # overwrite the params from config file with any written in this code above
        v['config'] = replace_in_dict(v['config'], v_usedfornaming)

        # experiment name
        v['exp_name'] = v['config']['logging']['log_dir'] \
            + '__'.join([v['config']['env']['name'], config['env']['task']] \
            + [k + '_' + str(v) for k,v in v_usedfornaming.items() if k not in ['metatrain_itr', 'horizon', 
            'n_candidates', 'dim_bias', 'name', 'task', 'dim_hidden', 'log_dir', 'summary_itr', 
            'print_itr', 'max_epochs', 'multi_input', 'meta_learn_lr']])

        # run
        run_experiment_lite(
            run,
            sync_s3_pkl=True,
            periodic_sync=True,
            variant=v,
            snapshot_mode="last",
            mode="local",
            use_cloudpickle=True,
            exp_name=v['exp_name'],
            use_gpu=use_gpu,
            pre_commands=[#"yes | pip install --upgrade pip",
                          "yes | pip install tensorflow=='1.4.1'",
                          "yes | pip install --upgrade cloudpickle"],
            seed=v['seed']
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_path = osp.join(osp.dirname(osp.realpath(__file__)), '../config/', 'config.yaml')
    parser.add_argument('-p', '--config_path', type=str, default=default_path,
                  help='directory for the config yaml file.')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()
    main(args.config_path, args.use_gpu) 

