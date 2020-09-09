'''
@Author: Mengdi Xu
@Email: 
@Date: 2020-03-23 00:00:07
@LastEditTime: 2020-04-15 22:32:41
@Description:
'''

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

"""Carpole SwingUp"""


register(
    id='CartPoleSwingUpEnv-v0',
    entry_point='cartpole_envs.envs:CartPoleSwingUpEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

