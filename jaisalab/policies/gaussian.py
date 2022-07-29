"""Semi-Implicit Actor (SIA). 

Implementation of SIA policy of: NEED TO PUT arxiv LINK. 
"""
import torch
from torch import nn

#garage
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.torch.distributions import TanhNormal
from garage.torch.modules.mlp_module import MLPModule

class GaussianPolicy(StochasticPolicy):
    """MLP whose outputs are fed into a Normal distribution..

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (EnvSpec): Environment specification.
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
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 min_log_std=1e-6, 
                 max_log_std=None,
                 layer_normalization=False,
                 name='GaussianPolicy'):
        super().__init__(env_spec, name)

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_sizes = hidden_sizes[:-1]
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._layer_normalization = layer_normalization

        #create policy architecture
        last_hidden_size = hidden_sizes[-1]

        self.base_fc = MLPModule(
                input_dim=self._obs_dim,
                output_dim=last_hidden_size,
                hidden_sizes=self._hidden_sizes,
                hidden_nonlinearity=self._hidden_nonlinearity,
                hidden_w_init=self._hidden_w_init,
                hidden_b_init=self._hidden_b_init,
                output_nonlinearity=self._output_nonlinearity,
                output_w_init=self._output_w_init,
                output_b_init=self._output_b_init,
                layer_normalization=self._layer_normalization)

        self.last_fc_mean = nn.Linear(last_hidden_size, self._action_dim)
        self.last_fc_log_std = nn.Linear(last_hidden_size, self._action_dim)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.Tensor: Batch of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        h = self.base_fc(observations) 
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(self._min_log_std, self._max_log_std).exp()

        dist = TanhNormal(mean, std)
        actions, pre_tanh_values = dist.rsample_with_pre_tanh_value()
        log_prob = dist.log_prob(actions, pre_tanh_value=pre_tanh_values)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return (dist._normal, dict(mean=dist.mean, 
                                   log_std=(dist.variance**.5).log(),
                                   log_prob=log_prob))
