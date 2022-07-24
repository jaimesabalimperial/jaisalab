"""
Author - Jaime Sabal
"""
import torch
import torch.functional as F
import math
import pdb
import torch.nn as nn
import numpy as np

class LinearNoise(nn.Linear):
    """Linear layer with Gaussian noise for Noisy Network 
    (https://arxiv.org/pdf/1706.10295.pdf) implementation for efficient exploration."""
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        # make the sigmas trainable
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        
        # Need tensor in buffer bu tnot as trainable parameter, like 'running_mean' for BatchNorm
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        # extra parameter for the bias and register buffer for the bias parameter
        if bias: 
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
    
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):
        """Initialise weights and biases of layer."""
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        """Forward call of noise layer."""
        # sample random noise in sigma weight buffer and bias buffer
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


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
        out = out.view(batch_size, self.episode_length, self.N, self._output_dim)
        
        return out, samples
    
    def get_quantiles(self, obs):
        """Summary of quantiles for batch of inputs."""
        with torch.no_grad():
            quantiles, _ = self.forward(obs)
        batch_quantiles = quantiles.mean(dim=0)

        return batch_quantiles

    def get_mean_std(self, obs):
        """Mean value and standard deviation of quantiles 
        for given inputs."""
        quantiles, _ = self.forward(obs)
        mean_values = quantiles.mean(dim=2).flatten(-2)
        std_values = quantiles.std(dim=2).flatten(-2)

        return mean_values, std_values
    

class Gaussian(object):
    def __init__(self, mu, rho, device=torch.device('cuda:0' if torch.cuda.is_available else 'cpu')):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.device = device

    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()