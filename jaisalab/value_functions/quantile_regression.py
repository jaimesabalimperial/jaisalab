#misc 
import torch 
import torch.nn as nn
import numpy as np

#jaisalab
from jaisalab.value_functions.modules import DistributionalModule
from jaisalab.utils.math import standard_deviation

#garage
from garage.torch.value_functions.value_function import ValueFunction


class QRValueFunction(ValueFunction):
    """Quantile Regression Value Function with Model.

    Paper: https://arxiv.org/pdf/1710.10044.pdf

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
                 name='QRValueFunction'):
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
        self.V_range = torch.tensor([self.Vmin + i*self.delta_z for i in range(self.N)])
        # set cumulative density
        self.cumulative_density = torch.FloatTensor((2 * np.arange(self.N) + 1) / (2.0 * self.N))
    
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
            V_dist, V_log_dist = self.module.forward(obs)

        mean = torch.matmul(V_dist, z_dist)
        std = standard_deviation(z_dist, V_dist)

        return mean, std
    
    def get_quantiles(self, obs):
        """Get quantile probabilities and values."""

        with torch.no_grad():
            V_dist = self.forward(obs, dist_output=True)
        
        V_dist = V_dist.flatten().tolist()

        return V_dist

    def compute_loss(self, obs, next_obs, rewards, 
                     masks, target_vf, gamma):
        """Compute quantile regression loss."""
        V_log_dist_pred = self.forward(obs, log_output=True) 
        V_target = target_vf.forward(next_obs, dist_output=True) #calculate value using target network

        m = torch.zeros(*obs.shape[:-1], self.N)
        for j in range(self.N):
            T_zj = torch.clamp(rewards + gamma * (1-masks) * (self.Vmin + j*self.delta_z), min = self.Vmin, max = self.Vmax)
            bj = (T_zj - self.Vmin)/self.delta_z
            l = bj.floor().long().unsqueeze(1)
            u = bj.ceil().long().unsqueeze(1)

            V_narrowed = torch.narrow(V_target, -1, j, 1)
            mask_Q_l = torch.zeros(m.size())
            mask_Q_l.scatter_(1, l, V_narrowed.squeeze(1))
            mask_Q_u = torch.zeros(m.size())
            mask_Q_u.scatter_(1, u, V_narrowed.squeeze(1))
            m += torch.matmul((u.float() + (l == u).float()-bj.float()), mask_Q_l)
            m += torch.matmul((-l.float()+bj.float()), mask_Q_u)

        #calculate Huber loss
        loss = - torch.mean(torch.sum(torch.sum(torch.mul(V_log_dist_pred, m),-1),-1) / obs.shape[0])
        return loss
