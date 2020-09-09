# MPC
This folder contains the implementation of MPC algorithm and the evaluation on the Pendulum and CartPole environment

The implementation is mainly followed in this paper [here](https://ieeexplore.ieee.org/abstract/document/8463189)

All the hyper-parameters and experiment setting are stored in the ```config.yml``` file

All the results (figure and model) will be stored in the ```./storage``` folder by default

If you are not familiar with this environment, you can use the  `analyze_env()`  function in the `utils.py` to help you quickly understand the environment's state space, action space, reward range, etc.

### Requirements

* tqdm
* pytorch
* OpenAI gym (v0.10.8)
* [DartEnv](https://github.com/DartEnv/dart-env/wiki) (Optional if you do not use CartPole env)

#### Trouble shooting

1. If see `ImportError: libdart.so.6.3: cannot open shared object file: No such file or directory` after install DartEnv, try `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib` before import it.


### How to run

Simple run

```angularjs
python train.py
```
The script will load the configurations in the ```config.yml``` file and begin to train.

Or you can run

```angularjs
python test.py
```
It will test the trained model that specified in ```config.yml model_path```.

### Configuration explanation

In the ```config.yml``` file, there are 4 sets of configuration.

The `model_config`  part is the configuration of the parameters which determine the neural network architecture and the environment basis.

The `training_config` part is the configuration of the training process parameters.

The `dataset_config` part is the configuration of the dataset parameters.

The `mpc_config` part is the configuration of the MPC algorithm parameters.

The `exp_number` parameter in the `training_config` is the number of your experiment. The name of saved figure results in the `./storage` folder will be determined by this parameter.

If you want to train your model from scratch, then set the `load_model` parameter to `False`. If set to `True`, the trainer will load the model from `model_path`.
