# CASRL

Context-aware safe reinforcement learning in non-stationary environments

<img src="assets/cp_short.gif?raw=true" width="45%"> 
<img src="assets/cp_long.gif?raw=true" width="45%"> 

<img src="assets/hc_forward.gif?raw=true" width="45%"> 
<img src="assets/hc_rotate.gif?raw=true" width="45%"> 

### Python3 requirements

```
scipy
matplotlib
pandas
gym==0.17.2
torch==1.5.1
```

To install the dependencies:
```
pip install -r requirements.txt
```

### Running experiments on Cartpole-swingup
```
python casrl_cartpole.py
```
Parameters are stored in in ./config/config_cartpole.yml

### Running experiments on Merge
```
python casrl_merge.py
```
Parameters are stored in in ./config/config_merge.yml

### Running experiments on Healthcare
First install my [fork](https://github.com/baimingc/assistive-gym) of assistive_gym environment. Then run
```
python casrl_healthcare.py
```
Parameters are stored in in ./config/config_healthcare.yml