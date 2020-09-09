<!--
 * @Author: Mengdi Xu, Wenhao Ding
 * @Email: 
 * @Date: 2020-04-01 14:57:24
 * @LastEditTime: 2020-04-17 20:34:08
 * @Description: 
 -->

# DPGP-MBRL

Use Dirichlet Process Gaussian Process for Model-based Refinement Learning.

### Python3 requirements

```
torch
gpytorch
gym
mujoco_py
matplotlib
numpy
scipy
loguru
```


### Running experiments on Cartpole
Two cartpole environments are provided, the first one is cartpole-stable and the second one is cartpole-swingup. Parameters are stored in in ./config/config_swingup.yml and ./config/config_stable.yml.
```angularjs
cd DPGP-MBRL/envs/cartpole-envs
pip install -e .
```

```angularjs
python3 train_on_swingup.py       # DPGP-MBRL with CartPole-swing-up
python3 train_on_swingup_NN.py    # NN-MBRL baseline with CartPole-swing-up (train with collected data)
python3 train_on_swingup_GrBAL.py # GrBAL baseline with CartPole-swing-up (train in meta-learning style)
python3 train_on_stable.py        # DPGP-MBRL with CartPole-v0
```


### Running experiments on FetchSlide
To use FetchSlide environment, you need to install mujoco_py first. Parameters are stored in in ./config/config_fetchslide.yml. We modified the FetchSlide-v1 environment of gym.robotics.
```angularjs
cd DPGP-MBRL/envs/fetchslide-env
pip install -e .
```

```angularjs
python3 train_on_fetchslide.py    # DPGP-MBRL with FetchSlide-v1
```


### Running experiments on Highway
Parameters are stored in in ./config/config_highway.yml.
```angularjs
cd DPGP-MBRL/envs/highway-env
pip install -e .
```

```angularjs
python3 test_highway.py           # Check roundabout with aggressive or defensive mode
```


### TODO:
* ~~Merge clusters~~
* ~~record scenatio transition matrix~~
* ~~Change Cartpole action space to continuous space~~
* ~~Sparse Gaussian Process Regression~~
* ~~integrate with MPC along traning~~, may still need to tune parameters
* ~~overall pipeline of MBRL~~
* ~~Test the SparseGP model~~
* ~~Make SingleGP and SingleSparseGP as our baselines~~
* ~~modify the official code of GrBAL (https://github.com/iclavera/learning_to_adapt) to be our baseline~~
* ~~check data point richness~~
* Design and creat highway scenarios
* Test on highway scenarios
* ~~Fix the bug of DPMixture: when a new component is still in the burnin stage, a newer component appears. This will terminate the program with a out-of-index error.~~
* Add one function: when the data index changes to an existing component, we also need to do the merge operation.
* Test on FetchSlide environment


### Note:
* In GPytorch, it's too expensive to check wheather a kernel matrix is positive-definite or not in practice (https://github.com/cornellius-gp/gpytorch/issues/1037). However, the data collected from gym-cartpole is deterministic, which will sometimes make the kernel matrix not positive-definite. Our solution is reseting the predict variance to the constraint value of noise_covar of likelihood.

* ~~In GPytorch, the prediction stage should be likelihood(model(test_x)), however, the likelihood() function will make a poor dynamic prediction, so we delete it for now. The function of likelihood() is adding a noise variance to the predicting variance.~~ ~~After tuning the prior parameters, we can use likelihood(model(test_x)) now.~~ We finaly decide to remove likelihood() in prediction, because adding this term will make the training very unstable.

* After training a long time, the linear_cg optimization cannot converge in 1000 (pre-defined value) iterations, then NaN will occur and termite the progress. We cannot solve it for now because the data size should not always accumulate. (Our final version should be DP-SparseGP)

* A good issue about the noisefree setting in GPytorch: https://github.com/cornellius-gp/gpytorch/issues/728

* We use torch.float64 as the date type to avoid numerical problems.