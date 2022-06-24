"""Constrained Policy Optimization using PyTorch."""
import torch

from garage.torch._functions import zero_optim_grads
from garage.torch.algos import VPG
from garage.torch.optimizers import OptimizerWrapper

from jaisalab.optimizers import ConjugateConstraintOptimizer
from jaisalab.utils import to_device

from copy import deepcopy
import numpy as np

"""
1). Differences between CPO and TRPO: 
    - Implement cost function to 



"""

class CPO(VPG):
    """Constrained Policy Optimization (CPO).

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        sampler (garage.sampler.Sampler): Sampler.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 sampler,
                 policy_optimizer=None,
                 vf_optimizer=None,
                 safety_constraint=None,
                 num_train_per_epoch=1,
                 discount=0.99,
                 gae_lambda=0.98,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy', 
                 grad_norm=False):

        if policy_optimizer is None:
            policy_optimizer = OptimizerWrapper(
                (ConjugateConstraintOptimizer, dict(max_kl=0.01)),
                policy)
        if vf_optimizer is None:
            vf_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=2.5e-4)),
                value_function,
                max_optimization_epochs=10,
                minibatch_size=64)
        if safety_constraint is None:
            self.safety_constraint = None
        
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.grad_norm = grad_norm

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         sampler=sampler,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         num_train_per_epoch=num_train_per_epoch,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method)

    def _compute_constraint_value(self, costs, masks):
        """Compute constraint value."""
        costs, masks = to_device(torch.device('cpu'), costs, masks)
        constraint_value = torch.tensor(0)
        
        j = 1
        traj_num = 1
        for i in range(costs.size(0)):
            constraint_value = constraint_value + costs[i] * self._discount**(j-1)
            
            if masks[i] == 0:
                j = 1 #reset
                traj_num = traj_num + 1
            else: 
                j = j+1
                
        constraint_value = constraint_value/traj_num
        constraint_value = to_device(self._device, constraint_value)
        return constraint_value

    def _compute_objective(self, advantages, obs, actions):
        r"""Compute objective value.

        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.

        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.

        """
        with torch.no_grad():
            old_ll = self._old_policy(obs)[0].log_prob(actions)

        new_ll = self.policy(obs)[0].log_prob(actions)
        likelihood_ratio = torch.exp(new_ll - old_ll)

        # Calculate surrogate
        surrogate = advantages * likelihood_ratio 

        return surrogate
    

    def _compute_cost_loss_with_adv(self, obs, actions, rewards, cost_advantages):
        r"""Compute mean value of loss.

        --> Only difference with _compute_loss_with_adv() method is that the cost 
        is defined as a minimisation objective while returns are trying to be 
        maximised.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
            cost_advantages (torch.Tensor): Advantage value at each step for costs
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of objective.

        """
        objectives = self._compute_objective(cost_advantages, obs, actions, rewards)

        if self._entropy_regularzied:
            policy_entropies = self._compute_policy_entropy(obs)
            objectives += self._policy_ent_coeff * policy_entropies

        return objectives.mean()
    
    def _get_loss_grad(self, obs, actions, rewards, advantages):
        """"""
        loss = self._compute_loss_with_adv(obs, actions, rewards, advantages)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach() #g  
        if self.grad_norm == True:
            loss_grad = loss_grad/torch.norm(loss_grad)
        return loss, loss_grad
    
    def _get_cost_loss_grad(self, obs, actions, rewards, cost_advantages):
        """"""
        cost_loss = self._compute_cost_loss_with_adv(obs, actions, rewards, cost_advantages)
        cost_grads = torch.autograd.grad(cost_loss, self.policy.parameters())
        cost_loss_grad = torch.cat([grad.view(-1) for grad in cost_grads]).detach() #g 
        cost_loss_grad = cost_loss_grad/torch.norm(cost_loss_grad) 
        return cost_loss, cost_loss_grad

    def _train_policy(self, obs, actions, rewards, advantages, cost_advantages):
        r"""Train the policy.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated mean scalar value of policy loss (float).

        """
        # pylint: disable=protected-access
        zero_optim_grads(self._policy_optimizer._optimizer)
        loss, loss_grad = self._get_loss_grad(obs, actions, rewards, advantages)
        cost_loss, cost_loss_grad = self._get_cost_loss_grad(self, obs, actions, rewards, cost_advantages)
        #loss.backward()
        #cost_loss.backward()

        constraint_value = self._compute_constraint_value(costs, masks)

        self._policy_optimizer.step(
            f_loss=lambda: self._compute_loss_with_adv(obs, actions, rewards,
                                                       advantages),
            f_cost=lambda: self._compute_cost_loss_with_adv(obs, actions, rewards,
                                                       cost_advantages),                                           
            f_constraint=lambda: self._compute_kl_constraint(obs), 
            loss_grad=loss_grad, 
            cost_loss_grad=cost_loss_grad, 
            )

        return loss, cost_loss