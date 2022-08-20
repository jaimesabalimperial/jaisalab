#misc 
import torch 
import torch.nn as nn
import numpy as np

#jaisalab
from jaisalab.value_functions.modules import DistributionalModule
from jaisalab.utils.math import calc_mean_std, log_prob

#garage
from garage.torch.value_functions.value_function import ValueFunction

class QRValueFunction(ValueFunction):
    """Quantile Regression Value Function with Model. We offer 
    a slight modification in terms of the value function, where instead 
    of calculating the quantile Huber loss we use a surrogate loss in 
    terms of the log likelihood of obtaining the observed returns with 
    the current predictions of the mean and standard deviations from the 
    estimated quantile distribution.

    Paper: https://arxiv.org/pdf/1710.10044.pdf

    Args:
        env_spec (EnvSpec): Environment specification.
        Vmin (int, float): Minimum of value range. 
        Vmax (int, float): Maximum of value range. 
        N (int): Number of quantiles to estimate. 
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): The name of the value function.

    """
    def __init__(self,
                 env_spec,
                 N, 
                 Vmin,
                 Vmax,
                 max_cost=None, #only relevant for safety baseline
                 tolerance=0.01, 
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='QRValueFunction'):
        super().__init__(env_spec, name)

        input_dim = env_spec.observation_space.flat_dim
        output_dim = 1
        self.N = N
        self.max_cost = max_cost
        self.tolerance = tolerance
    
        self.module = DistributionalModule(
                                input_dim=input_dim,
                                output_dim=output_dim,
                                N=N,
                                hidden_sizes=hidden_sizes,
                                hidden_nonlinearity=hidden_nonlinearity,
                                hidden_w_init=hidden_w_init,
                                hidden_b_init=hidden_b_init,
                                output_nonlinearity=output_nonlinearity,
                                output_w_init=output_w_init,
                                output_b_init=output_b_init,
                                layer_normalization=layer_normalization)
    
        #environment-specific
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax-Vmin)/(N-1)
        self.V_range = torch.tensor([self.Vmin + i*self.delta_z for i in range(self.N)])
    
    def forward(self, obs, log_output=False, dist_output=False):
        """Forward call to model."""
        #retrieve quantile values to estimate probabilities for
        z_dist = self.V_range.repeat(*obs.shape[:-1], 1)
        z_dist = torch.unsqueeze(z_dist, -1).float()

        #V_dist.shape = (obs.shape[:-1], 1, N); z_dist.shape = (obs.shape[:-1], N, 1)
        #Calculate mean value of quantiles
        V_dist, V_log_dist = self.module.forward(obs)

        if log_output:
            return V_log_dist
        elif dist_output: 
            return V_dist
        else: 
            V = torch.matmul(V_dist, z_dist).view(obs.shape[:-1])
            return V
        
    def get_mean_std(self, obs):
        """Get mean and standard deviation of quantile distribution."""
        #retrieve quantile values to estimate probabilities for
        z_dist = self.V_range.repeat(*obs.shape[:-1], 1)
        z_dist = torch.unsqueeze(z_dist, -1).float()

        with torch.no_grad():
            V_dist = self.forward(obs, dist_output=True)

        mean, std = calc_mean_std(V_dist, z_dist) #get mean and standard deviation

        return mean, std
    
    def get_quantiles(self, obs):
        """Get quantile probabilities and values."""

        with torch.no_grad():
            V_dist = self.forward(obs, dist_output=True)
        
        V_dist = V_dist.flatten().tolist()

        return V_dist

    def compute_loss(self, obs, returns):
        #retrieve quantile values to estimate probabilities for
        z_dist = self.V_range.repeat(*obs.shape[:-1], 1)
        z_dist = torch.unsqueeze(z_dist, -1).float()

        V_dist = self.forward(obs, dist_output=True)
        mean, std = calc_mean_std(V_dist, z_dist) #get mean and standard deviation
        ll = log_prob(returns.reshape(-1, 1), mean, std)
        loss = -ll.mean()
        return loss

class QuantileValueFunction(QRValueFunction):
    """"""

    