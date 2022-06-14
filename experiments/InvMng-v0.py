#misc
import os
import gym
from gym import spaces
import numpy as np
import random

#environment
from or_gym.envs.supply_chain import InvManagementBacklogEnv, InvManagementLostSalesEnv
import gym # already imported before
from gym.envs.registration import register
from gym import wrappers

#garage
from garage.np.baselines import LinearFeatureBaseline # <<<<<< requires restarting the runtime in colab after the 1st dependency installation above
from garage.envs import GymEnv
from garage.envs import normalize
from garage.tf.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage import Trainer
from garage.experiment.deterministic import set_seed
from dowel import logger, StdOutput
import garage

class NpWrapper(gym.ObservationWrapper):
    """Wrapper to convert observations to NumPy arrays. 
    Based on https://github.com/openai/gym/blob/5404b39d06f72012f562ec41f60734bd4b5ceb4b/gym/wrappers/dict.py
    """
    def observation(self, observation):
        obs = np.array(observation).astype('int')
        return obs

def env_setup(env):
    # set up the dowel logger
    log_dir = os.path.join(os.getcwd(), 'data')
    ctxt = garage.experiment.SnapshotConfig(snapshot_dir=log_dir,
                                            snapshot_mode='last',
                                            snapshot_gap=1)

    # log to stdout
    logger.add_output(StdOutput())

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


