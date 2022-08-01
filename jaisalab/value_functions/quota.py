
#garage
from garage.torch.value_functions.value_function import ValueFunction

#torch 
import torch 
import torch.nn as nn

#jaisalab
from jaisalab.value_functions.modules import DistributionalModule


class QUOTAValueFunction(ValueFunction):
    """QUOTA Value Function with Model.

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
        

        def compute_loss(self, obs, returns, **kwargs):
            pass