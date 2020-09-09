import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc import logger
from rllab.misc.overrides import overrides

from mole.envs.mujoco_env import MujocoEnv

class HalfCheetahActionsEnv(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, task=None, reset_every_episode=False, reward=False, *args, **kwargs):

        self.reset_every_episode = reset_every_episode
        self.first = True
        self.id_torso = 0
        super(HalfCheetahActionsEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.init_body_mass = self.model.body_mass.copy()
        self.id_torso = self.model.body_names.index('torso')
        self._reward = reward
        self._action_bounds = self.action_bounds
        self.step_num=0

        if task in [None, 'None', 
                    'rand_sign', #random, fixed for full rollout
                    'op1', 'op2', 'signPlus', 'signNeg', #specified, fixed for full rollout
                    'signChanging1','signChanging2', 'signChanging3', #alternate between 2 options, change within rollout
                    'signChanging_rand100', 'signChanging_rand300', 'signChanging_rand500' #random, change within rollout
                    'magnitude1', 'rand_magnitude']:
            self.task = task
            self.action_sign = np.ones(self.action_space.shape)
            self.op1 = np.array([-1,-1,-1,1,1,1])
            self.op2 = np.array([1,1,1,-1,-1,-1])
            self.opA = np.array([1,1,-1,-1,-1,1])
            self.opB = np.array([-1,-1,1,1,1,-1])
            self.op_pos = np.array([1,1,1,1,1,1])
            self.op_neg = np.array([-1,-1,-1,-1,-1,-1])
            self.sign=1
        else:
            raise NameError

        if self.task=='signChanging1': #alternate betw A and B
            self.action_sign = self.opA
        if self.task=='signChanging2': #alternate betw pos and neg
            self.action_sign = self.op_pos
        if self.task=='signChanging3': #alternate betw 1 and 2
            self.action_sign = self.op1

        self.my_state = []

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten(),
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action, currpathlength=-7):

        if currpathlength>=0:
            self.step_num= currpathlength

        if self.step_num>0:

            # alternate between 2 things, every 100 steps
            if self.task in ['signChanging1', 'signChanging2', 'signChanging3']:
                if self.step_num%100==0:
                    self.reset_task()

            # change randomly every 300 steps
            if self.task == 'signChanging_rand100':
                if self.step_num%100==0:
                    self.reset_task()

        # s,a --> s'
        curr_obs = self.get_current_obs()
        action = self.action_sign * action
        action = np.clip(action, *self._action_bounds)
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        
        # reward
        reward = 0
        if self._reward:
            reward = self.get_reward(curr_obs, next_obs, action, single=True)
        done = False
        return Step(next_obs, reward, done)

    def get_reward(self, observations, next_observations, actions, single=False):
        
        if single:
            rew = self.sign * (next_observations[-3] - observations[-3])/self.model.opt.timestep
            return rew
        else:
            rews = self.sign * (next_observations[:,-3] - observations[:,-3])/self.model.opt.timestep
            return rews

    def reset_mujoco(self, init_state=None):
        if init_state is None:
            super(HalfCheetahActionsEnv, self).reset_mujoco()
        else:
            super(HalfCheetahActionsEnv, self).reset_mujoco(init_state=init_state.copy())
        if self.reset_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False

        self.my_state = [self.model.data.qpos.copy(),self.model.data.qvel.copy(),
                        self.model.data.qacc.copy(),self.model.data.ctrl.copy()]

    def reset_task(self, value=None):

        if self.task == 'rand_sign':
            #randomly sample 0 or 1
            self.action_sign = np.random.choice(2, self.action_space.shape)
            #turn the 0 into a -1
            self.action_sign[self.action_sign==0]=-1

        ##########################

        elif self.task == 'op1':
            self.action_sign = self.op1

        elif self.task == 'op2':
            self.action_sign = self.op2

        elif self.task == 'signPlus':
            self.action_sign = self.op_pos

        elif self.task == 'signNeg':
            self.action_sign = self.op_neg

        ##########################

        elif self.task == 'signChanging1':
            if np.all(self.action_sign == self.opA):
                self.action_sign = self.opB
            else:
                self.action_sign = self.opA

        elif self.task == 'signChanging2':
            
            if np.all(self.action_sign == self.op_pos):
                self.action_sign = self.op_neg
            else:
                self.action_sign = self.op_pos

        elif self.task == 'signChanging3':
            
            if np.all(self.action_sign == self.op1):
                self.action_sign = self.op2
            else:
                self.action_sign = self.op1

        ##########################

        elif self.task == 'signChanging_rand300':
            #randomly sample -1/1 action signs
            self.action_sign = np.random.choice(2, self.action_space.shape)
            self.action_sign[self.action_sign==0]=-1

        ##########################

        elif self.task == 'rand_magnitude':
            self.action_sign = np.random.uniform(0.05, 2, self.action_space.shape[0])

        elif self.task == 'magnitude1':
            self.action_sign = [0.05, 0.1, 3, 3, 0.05, 2.0]

        ##########################

        elif self.task is None:
            pass
        elif self.task == 'None':
            pass
        else:
            raise NotImplementedError

        self.model.forward()

    def __getstate__(self):
        state = super(HalfCheetahActionsEnv, self).__getstate__()
        state['task'] = self.task
        state['reset_every_episode'] = self.reset_every_episode
        state['reward'] = self._reward
        return state

    def __setstate__(self, d):
        super(HalfCheetahActionsEnv, self).__setstate__(d)
        self.task = d['task']
        self.reset_every_episode = d['reset_every_episode']
        self._reward = d.get('reward', True)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))




