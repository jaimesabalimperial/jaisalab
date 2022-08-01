#misc 
import torch 
import torch.nn as nn
import numpy as np

#jaisalab
from jaisalab.value_functions.modules import DistributionalModule

#garage
from garage.torch.value_functions.value_function import ValueFunction


class QUOTAValueFunction(ValueFunction):
    """QUOTA Value Function with Model.

    Paper: https://arxiv.org/pdf/1811.02073.pdf

    Args:
        env_spec (EnvSpec): Environment specification.
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
                 Vmin=-800,
                 Vmax=800,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='QUOTAValueFunction'):
        super().__init__(env_spec, name)

        input_dim = env_spec.observation_space.flat_dim
        output_dim = 1
        self.N = N
    
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
        self.V_range = [self.Vmin + i*self.delta_z for i in range(self.N)]
        self.last_log_dist = None
    
    def forward(self, obs):
        """Forward call to model."""
        z_dist = torch.from_numpy(np.array([self.V_range])).repeat(*obs.shape[:-1], 1)
        z_dist = torch.unsqueeze(z_dist, -1).float()

        #V_dist.shape = (batch_size, 1, N); z_dist.shape = (batch_size, N)
        V_dist, V_log_dist = self.module.forward(obs)
        V = torch.matmul(V_dist, z_dist).view(obs.shape[:-1])
        self.last_log_dist = V_log_dist
        return V

    def compute_loss(self, obs, next_obs, actions, returns):
        """Compute loss."""
        z_dist = torch.from_numpy(np.array([[self.Vmin + i*self.delta_z for i in range(self.N)]]*obs.shape[0]))
        z_dist = torch.unsqueeze(z_dist, 2).float()
        print('z_dist = ', z_dist.shape)
        print('obs = ', obs.shape)
        print('next_obs = ',next_obs.shape)
        print('actions = ',actions.shape)
        print('returns = ',returns.shape)
        raise Exception
