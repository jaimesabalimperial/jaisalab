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
    """Helper function for setting up an OpenAI Gym environment
       under the garage framework.
    """
    if isinstance(env, InvManagementBacklogEnv):
        env_name = 'InvManagement-v0'
    elif isinstance(env, InvManagementLostSalesEnv):
        env_name = 'InvManagement-v1'

    #wrap as per garage framework
    env = gym.make(env_name)
    env = NpWrapper(env)
    env = normalize(GymEnv(env))

    return env