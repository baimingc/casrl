#!/bin/bash

filename1='TODO'

#####################################
#signChanging_rand300

python launch_maml_sim_policy.py -f $filename1 -t signChanging_rand300 -n 5 -l 2000 --model_num 15 --num_rollouts 5
python launch_maml_sim_policy.py -f $filename1 -t signChanging_rand300 -n 3 -l 2000 --model_num 15 --num_rollouts 5
python launch_maml_sim_policy.py -f $filename1 -t signChanging_rand300 -n 2 -l 2000 --model_num 15 --num_rollouts 5

#####################################
#signChanging1

python launch_maml_sim_policy.py -f $filename1 -t signChanging1 -n 5 -l 2000 --model_num 15 --num_rollouts 5
python launch_maml_sim_policy.py -f $filename1 -t signChanging1 -n 3 -l 2000 --model_num 15 --num_rollouts 5
python launch_maml_sim_policy.py -f $filename1 -t signChanging1 -n 2 -l 2000 --model_num 15 --num_rollouts 5

#####################################
#signNeg

python launch_maml_sim_policy.py -f $filename1 -t signNeg -n 5 -l 2000 --model_num 15 --num_rollouts 5
python launch_maml_sim_policy.py -f $filename1 -t signNeg -n 3 -l 2000 --model_num 15 --num_rollouts 5
python launch_maml_sim_policy.py -f $filename1 -t signNeg -n 2 -l 2000 --model_num 15 --num_rollouts 5
