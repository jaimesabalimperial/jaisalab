#torch
import torch
import torch.nn as nn

#misc
import numpy as np

#garage
from garage.torch.value_functions.value_function import ValueFunction
from garage.torch import NonLinearity

#jaisalab
from jaisalab.utils import LinearNoise

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

class IQNModule(nn.Module): 
    """Base module for Implicit Quantile Network implementation."""
    def __init__(self,
                 input_dim,
                 action_size,
                 N,
                 layer_size=128, 
                 n_cos=64, 
                 *, 
                 dueling=False, 
                 noisy=False):
        super().__init__()

        self._input_dim = input_dim
        self._action_size = action_size
        self._n_cos = n_cos
        self.dueling = dueling 
        self.noisy = noisy
        self.N = N
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        layer = LinearNoise if noisy else nn.Linear #chose type of fully connected layer
        
        #define network architecture
        self.head = nn.Linear(self._input_shape[0], layer_size) 
        self.cos_embedding = nn.Linear(self._n_cos, layer_size)
        self.fc1 = layer(layer_size, layer_size)
        self.cos_layer_out = layer_size

        #dueling modification
        if self.dueling:
            self.advantage = layer(layer_size, action_size)
            self.value = layer(layer_size, 1)
        else:
            self.fc2 = layer(layer_size, action_size)  
    
    def get_cosine_values(self, batch_size):
        """Calculates cosine values for sample embedding."""
        samples = torch.rand(batch_size, self.N).unsqueeze(-1).to(self.device) #(batch_size, n_tau, 1)  .to(self.device)
        cos_values = torch.cos(samples*self.pis)

        return cos_values, samples

    def forward(self, inputs):
        """Quantile calculation based on the number of samples
        used in estimating the loss N (tau in paper).
        """
        batch_size = inputs.shape[0]

        x = torch.relu(self.head(inputs))
        cos_values, samples = self.calc_cos(batch_size, self.N) # cos shape (batch, num_tau, layer_size)
        cos_values = cos_values.view(batch_size*self.N, self._n_cos)
        cos_x = torch.relu(self.cos_embedding(cos_values)).view(batch_size, self.N, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*self.N, self.cos_layer_out)
        
        x = torch.relu(self.fc1(x))
        if self.dueling:
            advantage = self.advantage(x)
            value = self.value(x)
            out = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            out = self.fc2(x)
        
        return out.view(batch_size, self.N, self.action_size), samples
    
    def get_qvalues(self, inputs):
        quantiles, _ = self.forward(inputs, self.N)
        actions = quantiles.mean(dim=1)
        return actions  


class IQNValueFunction(ValueFunction): 
    """Implicit Quantile Network implementation. 
    
    Paper: https://arxiv.org/pdf/1806.06923v1.pdf
    
    Args: 
        env_spec (garage.EnvSpec) - Environment specification.
        layer_size (int) - Number of neurons in hidden layers. 
        N (int) - Number of samples used to estimate the loss.
        dueling (bool) - Wether to implement the Dueling Network Architecture. 
        noisy (bool) - Wether to add Gaussian noise to the linear layers
                       of the network (Noisy Network Architecture). 
    """
    def __init__(self, 
                env_spec, 
                layer_size,
                N, 
                dueling=False, 
                noisy=False,
                name='IQN'):
        super().__init__(env_spec, name)

        self._env_spec = env_spec 
        self._input_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._layer_size = layer_size

        self.N = N #number of samples used to estimate the loss
        self.n_cos = 64 #cosine embedding dimension as in the paper

        #define IQN module for forward pass
        self.module = IQNModule(input_dim=self._input_dim, 
                                action_dim=self._action_dim, 
                                N=self.N, 
                                layer_size=self._layer_size, 
                                n_cos=self.n_cos, 
                                dueling=dueling, 
                                noisy=noisy)
    
    def compute_loss(self, obs, returns):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
        #TODO: NEED TO IMPLEMENT HUBER LOSS
        pass

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
        return self.module(obs)

