"""
Taken directly from: https://github.com/huawei-noah/HEBO/blob/master/SAUTE/envs/wrappers/safe_env.py

Citation: 
@article{sootla2022saute,
  title={SAUTE RL: Almost Surely Safe Reinforcement Learning Using State Augmentation},
  author={Sootla, Aivar and Cowen-Rivers, Alexander I and Jafferjee, Taher and Wang, 
          Ziyan and Mguni, David and Wang, Jun and Bou-Ammar, Haitham},
  journal={arXiv preprint arXiv:2202.06558},
  year={2022}
}
"""

from gym import Env
import numpy as np


class SafeEnv(Env):
    """Safe environment wrapper."""
    def step(self, action:np.ndarray) -> np.ndarray:
        state = self._get_state()
        next_state, reward, done, info = super().step(action)
        info['cost'] = self._safety_cost_fn(state, action, next_state)
        return next_state, reward, done, info

    def _get_state(self):
        """Returns current state. Uses _get_obs() method if it is implemented."""
        if hasattr(self, "_get_obs"):
            return self._get_obs()
        else:
            raise NotImplementedError("Please implement _get_obs method returning the current state")                     

    def _safety_cost_fn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> np.ndarray:        
        """Returns current safety cost."""
        raise NotImplementedError("Please implement _safety_cost_fn method returning the current safety cost")    