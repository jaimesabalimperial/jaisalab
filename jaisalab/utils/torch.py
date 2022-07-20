"""
Author - Jaime Sabal
"""
import torch
import torch.functional as F
import math
import pdb
import torch.nn as nn

def to_device(device, *args):
    return [x.to(device) for x in args]

"""Functions to initialize weights of PyTorch models differently."""
def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

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




