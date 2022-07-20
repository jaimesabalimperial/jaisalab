#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.independent import Independent

#misc
import numpy as np
import math
import abc

#garage
from garage.torch.value_functions.value_function import ValueFunction
from garage.torch.distributions import TanhNormal


class IQNModule(nn.Module): 
    """Base of IQNModel.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
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
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 *,
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_std = learn_std
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._norm_dist_class = normal_distribution_cls

        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        self._init_std = torch.Tensor([init_std]).log()
        log_std = torch.Tensor([init_std] * output_dim).log()
        if self._learn_std:
            self._log_std = torch.nn.Parameter(log_std)
        else:
            self._log_std = log_std
            self.register_buffer('log_std', self._log_std)

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
            self.register_buffer('min_std_param', self._min_std_param)
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()
            self.register_buffer('max_std_param', self._max_std_param)

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._log_std, torch.nn.Parameter):
            self._log_std = buffers['log_std']
        self._min_std_param = buffers['min_std_param']
        self._max_std_param = buffers['max_std_param']

    @abc.abstractmethod
    def _get_mean_and_log_std(self, *inputs):
        pass

    def forward(self, *inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.

        """
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()
        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist




class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # extra parameter for the bias and register buffer for the bias parameter
        if bias: 
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
    
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    
    def forward(self, input):
        # sample random noise in sigma weight buffer and bias buffer
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class IQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, n_step, seed, N, dueling=False, noisy=False, device="cuda:0"):
        super(IQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.state_dim = len(self.input_shape)
        self.action_size = action_size
        self.N = N  
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1,self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.dueling = dueling
        self.device = device
        if noisy:
            layer = NoisyLinear
        else:
            layer = nn.Linear

        # Network Architecture
        if self.state_dim == 3:
            self.head = nn.Sequential(
                nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            )#.apply() #weight init
            self.cos_embedding = nn.Linear(self.n_cos, self.calc_input_layer())
            self.ff_1 = layer(self.calc_input_layer(), layer_size)
            self.cos_layer_out = self.calc_input_layer()

        else:
            self.head = nn.Linear(self.input_shape[0], layer_size) 
            self.cos_embedding = nn.Linear(self.n_cos, layer_size)
            self.ff_1 = layer(layer_size, layer_size)
            self.cos_layer_out = layer_size
        if dueling:
            self.advantage = layer(layer_size, action_size)
            self.value = layer(layer_size, 1)
            #weight_init([self.head_1, self.ff_1])
        else:
            self.ff_2 = layer(layer_size, action_size)    
            #weight_init([self.head_1, self.ff_1])

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.head(x)
        return x.flatten().shape[0]
        
    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device) #(batch_size, n_tau, 1)  .to(self.device)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, num_tau=8):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]
        
        x = torch.relu(self.head(input))
        if self.state_dim == 3: x = x.view(input.size(0), -1)
        cos, taus = self.calc_cos(batch_size, num_tau) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.cos_layer_out)
        
        x = torch.relu(self.ff_1(x))
        if self.dueling:
            advantage = self.advantage(x)
            value = self.value(x)
            out = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            out = self.ff_2(x)
        
        return out.view(batch_size, num_tau, self.action_size), taus
    
    def get_qvalues(self, inputs):
        quantiles, _ = self.forward(inputs, self.N)
        actions = quantiles.mean(dim=1)
        return actions  