#misc
import gym
import numpy as np

#garage
from garage.envs import normalize, GymEnv


"""Wrapper class that converts gym.Env into GymEnv."""
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
    #wrap as per garage framework
    if isinstance(env, str):
        env = gym.make(env)

    env = NpWrapper(env)
    env = normalize(GymEnv(env))

    return env
