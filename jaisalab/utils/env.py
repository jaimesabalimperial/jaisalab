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
    env = normalize(GymEnv(env, max_episode_length=env.spec.max_episode_length))

    return env

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                        type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")
