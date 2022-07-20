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
    """Base of IQNModel.
    """

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

        layer = LinearNoise if noisy else nn.Linear #chose type of fully connected layer
        
        #define network architecture
        self.head = nn.Linear(self._input_shape[0], layer_size) 
        self.cos_embedding = nn.Linear(self._n_cos, layer_size)
        self.ff_1 = layer(layer_size, layer_size)
        self.cos_layer_out = layer_size

        #dueling modification
        if self.dueling:
            self.advantage = layer(layer_size, action_size)
            self.value = layer(layer_size, 1)
        else:
            self.ff_2 = layer(layer_size, action_size)  
        

class IQN(ValueFunction): 
    """Implicit Quantile Networks implementation. 
    
    Paper: https://arxiv.org/pdf/1806.06923v1.pdf
    
    Args: 
        env_spec (garage.EnvSpec) - Environment specification.
        layer_size (int) - Number of neurons in hidden layers. 
        N (int) - Number of samples used to estimate the loss.
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
        self.module = IQNModule(self._input_dim, 
                                self._action_dim, 
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
        #TODO: IMPLEMENT FORWARD CALL TO IQN
        pass

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
            layer = LinearNoise
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