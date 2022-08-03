#torch
from logging import exception
import torch
import torch.nn as nn

#misc
import numpy as np
from numbers import Number
import math

#garage
from garage.torch.value_functions.value_function import ValueFunction

#jaisalab
from jaisalab.value_functions.modules import IQNModule

#TODO: NEED TO FIX

class IQNValueFunction(ValueFunction): 
    """Implicit Quantile Network implementation. A slight modification 
    is made in that we are learning the value function rather than the 
    Q-function as they do in the paper. We also use the mean of the log 
    probability of the observed returns as per the current return 
    distribution as the loss rather than the Huber loss (which requires
    learning in an off-policy manner). This is done to allow for 
    compatibility with on-policy policy gradient methods such as CPO, 
    TRPO, etc. 
    
    Paper: https://arxiv.org/pdf/1806.06923v1.pdf
    
    Args: 
        env_spec (garage.EnvSpec) - Environment specification.
        N (int) - Number of samples reparametrized from U([0,1]) to 
                  the quantile values of the return distribution. 
        layer_size (int) - Size of hidden layers in neural network.
        n_cos (int) - Number of inputs to the cosine embedding layer. 
        noisy (bool) - Wether to add Gaussian noise to the linear layers
                       of the network (Noisy Network Architecture). 
    """
    def __init__(self, 
                env_spec, 
                layer_size,
                N=50, 
                noisy=False,
                name='IQN'):
        super().__init__(env_spec, name)

        self._env_spec = env_spec 
        self._input_dim = env_spec.observation_space.flat_dim
        self._layer_size = layer_size
        self._episode_length = env_spec.max_episode_length

        self.N = N #number of samples used to estimate the loss
        self.n_cos = 64 #cosine embedding dimension as in the paper

        #define IQN module for forward pass
        self.module = IQNModule(input_dim=self._input_dim, 
                                episode_length=self._episode_length,
                                N=self.N, 
                                layer_size=self._layer_size, 
                                n_cos=self.n_cos, 
                                noisy=noisy)

    def get_quantiles(self, obs):
        """Get the mean value of the quantiles for a batch of observations."""
        return self.module.get_quantiles(obs)

    def get_mean_std(self, obs):
        """Retrieve mean and standard deviation of return distribution"""
        return self.module.get_mean_std(obs)
    
    def _log_prob(self, mean, std, value):
        """Log probability of obtaining a value given a distribution."""
        # compute the variance
        var = (std ** 2)
        log_scale = math.log(std) if isinstance(std, Number) else std.log()
        return -((value - mean) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    
    def _calculate_huber_loss(self, error, k=1.0):
        loss = torch.where(error.abs() <= k, 0.5 * error.pow(2), k * (error.abs() - 0.5 * k))
        return loss

    def compute_loss(self, obs, returns):
        r"""Compute mean loss using observations and returns.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).
        """
        mean, std = self.get_mean_std(obs)
        mean = torch.flatten(mean)
        std = torch.flatten(std)
        ll = self._log_prob(mean, std, returns.reshape(-1,1))
        loss = -ll.mean()

        return loss

    # pylint: disable=arguments-differ
    def forward(self, obs):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """
        quantiles, _ = self.module(obs)
        return quantiles.mean(dim=2).flatten(-2)

