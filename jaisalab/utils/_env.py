#misc
import gym
from gym import spaces
import numpy as np
import random
import torch

#environment
from or_gym.envs.supply_chain import InvManagementBacklogEnv, InvManagementLostSalesEnv
import gym # already imported before
from gym.envs.registration import register

#garage
import garage
from garage.envs import GymEnv
from garage.envs import normalize

class NpWrapper(gym.ObservationWrapper):
    """Wrapper to convert observations to NumPy arrays. 
    Based on https://github.com/openai/gym/blob/5404b39d06f72012f562ec41f60734bd4b5ceb4b/gym/wrappers/dict.py
    """
    def observation(self, observation):
        obs = np.array(observation).astype('int')
        return obs

def env_setup(env):

    if isinstance(env, InvManagementBacklogEnv):
        num = 0
    elif isinstance(env, InvManagementLostSalesEnv):
        num = 1

    #register env with openai-gym
    register(id=f'InvMng-v{num}', entry_point=env)

    # test registration was successful
    env = gym.make(f"InvMng-v{num}")
    env = NpWrapper(env)
    env = normalize(GymEnv(env))

    return env