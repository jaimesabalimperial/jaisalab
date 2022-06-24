"""Constrained Policy Optimization using PyTorch with the garage framework."""
import torch
from dowel import tabular

from garage.torch._functions import zero_optim_grads
from garage.torch.algos import VPG
from garage.torch.optimizers import OptimizerWrapper
from garage.torch import filter_valids
from garage.torch._functions import np_to_torch, zero_optim_grads
from garage import log_performance
from garage.np import discount_cumsum

from jaisalab.optimizers import ConjugateConstraintOptimizer
from jaisalab.utils import to_device
from jaisalab.safety_constraints import InventoryConstraints

from copy import deepcopy
import numpy as np


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
            self.safety_constraint = InventoryConstraints()
        else: 
            self.safety_constraint = safety_constraint
        
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

    def _compute_objective(self, advantages, obs, actions, rewards):
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
    

    def _compute_cost_loss_with_adv(self, obs, actions, costs, cost_advantages):
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
        objectives = self._compute_objective(cost_advantages, obs, actions, costs)

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

    def _train_once(self, itr, eps):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            eps (EpisodeBatch): A batch of collected paths.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """
        obs = np_to_torch(eps.padded_observations)
        rewards = np_to_torch(eps.padded_rewards)
        returns = np_to_torch(np.stack([discount_cumsum(reward, self.discount)
                              for reward in eps.padded_rewards]))
        valids = eps.lengths
        masks = np_to_torch(eps.env_infos["mask"])

        with torch.no_grad():
            baselines = self._value_function(obs)

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = np_to_torch(eps.observations)
        actions_flat = np_to_torch(eps.actions)
        rewards_flat = np_to_torch(eps.rewards)
        returns_flat = torch.cat(filter_valids(returns, valids))
        costs_flat = np_to_torch(self.safety_constraint.evaluate(eps))
        cost_advs_flat = None
        
        advs_flat = self._compute_advantage(rewards, valids, baselines)

        with torch.no_grad():
            policy_loss_before = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_before = self._value_function.compute_loss(
                obs_flat, returns_flat)
            kl_before = self._compute_kl_constraint(obs)

        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat, costs_flat, cost_advs_flat, masks)

        with torch.no_grad():
            policy_loss_after = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_after = self._value_function.compute_loss(
                obs_flat, returns_flat)
            kl_after = self._compute_kl_constraint(obs)
            policy_entropy = self._compute_policy_entropy(obs)

        with tabular.prefix(self.policy.name):
            tabular.record('/LossBefore', policy_loss_before.item())
            tabular.record('/LossAfter', policy_loss_after.item())
            tabular.record('/dLoss',
                           (policy_loss_before - policy_loss_after).item())
            tabular.record('/KLBefore', kl_before.item())
            tabular.record('/KL', kl_after.item())
            tabular.record('/Entropy', policy_entropy.mean().item())

        with tabular.prefix(self._value_function.name):
            tabular.record('/LossBefore', vf_loss_before.item())
            tabular.record('/LossAfter', vf_loss_after.item())
            tabular.record('/dLoss',
                           vf_loss_before.item() - vf_loss_after.item())

        self._old_policy.load_state_dict(self.policy.state_dict())

        undiscounted_returns = log_performance(itr,
                                               eps,
                                               discount=self._discount)
        return np.mean(undiscounted_returns)
    

    def _train(self, obs, actions, rewards, returns, advs, costs, cost_advs, masks):
        r"""Train the policy and value function with minibatch.

        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.

        """
        for dataset in self._policy_optimizer.get_minibatch(
                obs, actions, rewards, advs, costs, cost_advs, masks):
            self._train_policy(*dataset)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)

    def _train_policy(self, obs, actions, rewards, advantages, 
                      costs, cost_advantages, masks):
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
            constraint_value=constraint_value, 
            d_k=0)

        return loss, cost_loss