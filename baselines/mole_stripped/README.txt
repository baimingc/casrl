# mole

##############################

1) SETUP

	$ cd <path_to_here>
	$ conda env create -f environment.yml
	$ source activate mole
	$ export PYTHONPATH=<path_to_here>

Note: add the last line above to your ~/.bashrc file to prevent having to do this in every new terminal

##############################

2) TRAIN THETA* WITH META-LEARNING (maml on the dynamics model)

	$ cd mole/scripts
	$ CUDA_VISIBLE_DEVICES=0 python launch_maml_train.py --use_gpu

##############################

3) SIMULATE DIFFERENT RUN-TIME OPTIONS, starting from theta* model trained using the commands from step2:

	a) always reset from theta*:
		$ python launch_maml_sim_policy.py -f <filepath_to_saved> -t signChanging1 -n 2 -l 2000 --model_num 15 --num_rollouts 5
	b) always take a gradient step away from previous weight:
		$ python launch_maml_sim_policy.py -f <filepath_to_saved> -t signChanging1 -n 3 -l 2000 --model_num 15 --num_rollouts 5
	c) MOLE(ours):
		$ python launch_maml_sim_policy.py -f <filepath_to_saved> -t signChanging1 -n 5 -l 2000 --model_num 15 --num_rollouts 5

See scripts/run_cheetahac_tests1.sh for examples

Notes:
	-t is the task (See env file to see what the tasks are)
	-n is which option to use for weight updates (2,3,5)
	-l is the rollout length
	--render can be used to visualize it live (note that you can also visualize these afterward, with command shown below)

##############################

4) VISUALIZE A ROLLOUT that you created using the commands from step3:

	$ python visualize_rollout.py -f <filepath_to_saved> --speedup 20 -render -t signChanging1 -n 2 -r 4
	$ python visualize_rollout.py -f <filepath_to_saved> --speedup 20 -render -t signChanging1 -n 3 -r 4
	$ python visualize_rollout.py -f <filepath_to_saved> --speedup 20 -render -t signChanging1 -n 5 -r 4

Special note:
	You cannot use this visualize script when the task itself involves selecting random values (e.g., signChanging_rand500), because the task will not be set exactly the same during this playback as it was during the original run. You'll see the effect of this via the printouts that appear after the rollout is completed ('RECORDED' will not match 'ACHIEVED').

Notes:
	-t is the task (See env file to see what the tasks are)
	-n is which weight update option (2,3,5) you'd like to visualize
	-l is the rollout length
	-r is which rollout (0-4, since num_rollouts above was 5)
	-render is to visualize it