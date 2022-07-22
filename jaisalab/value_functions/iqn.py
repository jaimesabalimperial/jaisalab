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
from garage.torch import NonLinearity

#jaisalab
from jaisalab.utils import LinearNoise

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

class IQNModule(nn.Module): 
    """Base module for Implicit Quantile Network implementation.

    Args: 
        input_dim (tuple) - Shape of model inputs.
        N (int) - Number of samples reparametrized from U([0,1]) to 
                  the quantile values of the return distribution. 
        layer_size (int) - Size of hidden layers in neural network.
        n_cos (int) - Number of inputs to the cosine embedding layer. 
        noisy (bool) - Wether to add Gaussian noise to the linear layers
                of the network (Noisy Network Architecture). 

    """
    def __init__(self,
                 input_dim,
                 episode_length,
                 N,
                 layer_size=128, 
                 n_cos=64, 
                 *, 
                 noisy=False):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = 1
        self._n_cos = n_cos
        self.noisy = noisy
        self.N = N
        self.episode_length = episode_length
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Starting from 0 as in the paper 
        self.pis = torch.FloatTensor([[np.pi*i for i in range(1,self._n_cos+1)] \
                                       for _ in range(episode_length)]).view(1,episode_length,self._n_cos).to(self.device) 

        layer = LinearNoise if noisy else nn.Linear #choose type of fully connected layer
        
        #define network architecture
        self.head = nn.Linear(self._input_dim, layer_size) 
        self.cos_embedding = nn.Linear(self._n_cos, layer_size)
        self.fc1 = layer(layer_size, layer_size)
        self.cos_layer_out = layer_size
        self.fc2 = layer(layer_size, self._output_dim) #output one value for a given input 
    
    def get_cosine_values(self, batch_size):
        """Calculates cosine values for sample embedding."""
        samples = torch.rand(batch_size, self.episode_length, self.N).unsqueeze(-1).to(self.device) 
        cos_values = torch.cos(samples*self.pis.unsqueeze(2))

        return cos_values, samples

    def forward(self, obs):
        """Quantile calculation based on the number of samples
        used in estimating the loss N (tau in paper)."""
        batch_size = obs.shape[0]

        x = torch.relu(self.head(obs))
        cos_values, samples = self.get_cosine_values(batch_size) # cos shape (batch, num_tau, layer_size)
        cos_values = cos_values.view(batch_size*self.N, self.episode_length, self._n_cos)
        cos_x = torch.relu(self.cos_embedding(cos_values)).view(batch_size, self.N, self.episode_length, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*self.N, self.episode_length, self.cos_layer_out)

        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        
        return out.view(batch_size, self.episode_length, self.N, self._output_dim), samples
    
    def get_quantiles(self, obs):
        """Summary of quantiles for batch of inputs."""
        with torch.no_grad():
            quantiles, _ = self.forward(obs)
        batch_quantiles = quantiles.mean(dim=0)

        return batch_quantiles

    def get_mean_std(self, obs):
        """Mean value and standard deviation of quantiles 
        for given inputs."""
        with torch.no_grad():
            quantiles, _ = self.forward(obs)

        mean_values = quantiles.mean(dim=1)
        std_values = quantiles.std(dim=1)

        return mean_values, std_values
    

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
        ll = self._log_prob(mean, std, returns.reshape(-1, 1))
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

