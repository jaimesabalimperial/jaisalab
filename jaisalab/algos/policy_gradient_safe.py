"""
Implements policy gradient algorithms in environments where safety
is considered (i.e. where there is a cost associated with state-action pairs).

Author: Jaime Sabal Bermúdez
"""
import torch
from dowel import tabular
import copy
import inspect

#garage
from garage.torch.algos import VPG
from garage.torch import (compute_advantages, filter_valids, 
                          global_device, state_dict_to)
from garage.torch._functions import np_to_torch, zero_optim_grads
from garage.np import discount_cumsum

#jaisalab
from jaisalab.sampler.safe_worker import SafeWorker
from jaisalab.utils import log_performance
from jaisalab.safety_constraints import SoftInventoryConstraint, BaseConstraint
from jaisalab.sampler.sampler_safe import SamplerSafe
from jaisalab.value_functions import GaussianValueFunction

import numpy as np


class PolicyGradientSafe(VPG):
    """
    Policy Gradient base algorithm

    with optional data reuse and importance sampling,
    and exploration bonuses

    also with safety constraints

    Can use this as a base class for VPG, CPO, TRPO, etc. by picking appropriate optimizers / arguments

    for CPO: Use ConjugateConstraint optimizer with max_backtracks>1
    for TRPO: use ConjugateGradient optimizer with max_backtracks>1
    """
    def __init__(self,
                env_spec,
                policy,
                value_function,
                sampler,
                policy_optimizer=None,
                vf_optimizer=None,
                safety_constrained_optimizer=True,
                safety_constraint=None,
                safety_discount=1,
                safety_gae_lambda=1,
                center_safety_vals=True,
                num_train_per_epoch=1,
                step_size=0.01,
                discount=0.99,
                gae_lambda=0.98,
                center_adv=True,
                positive_adv=False,
                policy_ent_coeff=0.0,
                use_softplus_entropy=False,
                stop_entropy_gradient=False,
                entropy_method='no_entropy', 
                grad_norm=False, 
                is_saute = False):

        if safety_constraint is None:
            #by default use a Gaussian baseline and soft constraints for IMP
            safety_baseline = GaussianValueFunction(env_spec=env_spec,
                                                     hidden_sizes=(64, 64),
                                                     hidden_nonlinearity=torch.tanh,
                                                     output_nonlinearity=None)

            self.safety_constraint = SoftInventoryConstraint(baseline=safety_baseline)
        else: 
            if isinstance(safety_constraint, BaseConstraint):
                self.safety_constraint = safety_constraint
            else: 
                raise TypeError("Safety constraint has to inherit from BaseConstraint.")
        
        self._safety_optimizer = self.safety_constraint.baseline_optimizer
        self._safety_baseline = self.safety_constraint.baseline
        self.safety_constrained_optimizer = safety_constrained_optimizer
        self.step_size = step_size 
        self.safety_step_size = self.safety_constraint.safety_step

        if sampler is None: 
            self.sampler = SamplerSafe()
        else: 
            worker_class = sampler._factory._worker_class
            if not issubclass(worker_class, SafeWorker):
                raise TypeError("Worker class must be a jaisalab.sampler.SafeWorker object.")
            self.sampler = sampler

        self.grad_norm = grad_norm
        self.env_spec = env_spec
        self.sampler.algo = self
        self.safety_discount = safety_discount
        self.safety_gae_lambda = safety_gae_lambda
        self.center_safety_vals = center_safety_vals
        self._is_saute = is_saute

        #environment-specific attributes (change or ignore at will)
        if self.env_spec.observation_space.flat_dim == 33:
            self.initial_state = torch.tensor([100., 100., 200., 0., 0., 0., 0., 0., 0., 0., 
                                               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                                               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        #for SAUTE algorithms states are augmented
        elif self.env_spec.observation_space.flat_dim == 34:
            self.initial_state = torch.tensor([100., 100., 200., 0., 0., 0., 0., 0., 0., 0., 
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])


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

    def _compute_safety_advantage(self, safety_rewards, valids, safety_baselines):
        r"""Compute mean value of loss.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.

        Returns:
            torch.Tensor: Calculated advantage values given rewards and
                baselines with shape :math:`(N \dot [T], )`.

        """
        advantages = compute_advantages(self.safety_discount, self.safety_gae_lambda,
                                        self.max_episode_length, safety_baselines,
                                        safety_rewards)
        advantage_flat = torch.cat(filter_valids(advantages, valids))

        if self.center_safety_vals:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat
    
    def _get_grad(self, loss):
        """Get the gradient of the loss with respect to policy parameters."""
        grads = torch.autograd.grad(loss, self.policy.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach() #b

        return loss_grad
    
    def get_mean_std(self, vf, obs):
        """Get the mean and standard deviation of a state-value
        function after training on the latest observations."""
        mean_std_func = getattr(vf, 'get_mean_std', None)
        #check if there is a function to retrieve the mean and std
        if callable(mean_std_func):
            mean, stddev = mean_std_func(obs)
            mean = torch.flatten(mean).mean()
            stddev = torch.flatten(stddev).mean()
            return mean, stddev
        else: 
            return None, None
    
    def get_quantiles(self, vf, obs):
        """Get the quantiles of the state-value
        distribution after training on the latest observations."""
        quantiles_func = getattr(vf, 'get_quantiles', None)
        #check if there is a function to retrieve the mean and std
        if callable(quantiles_func):
            quantile_probs = quantiles_func(obs)
            return quantile_probs
        else: 
            return None, None
        
    def _train_once(self, itr, eps):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            eps (SafeEpisodeBatch): A batch of collected paths.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """
        obs = np_to_torch(eps.padded_observations)
        rewards = np_to_torch(eps.padded_rewards)
        returns = np_to_torch(np.stack([discount_cumsum(reward, self.discount)
                              for reward in eps.padded_rewards]))
        valids = eps.lengths

        with torch.no_grad():
            baselines = self._value_function(obs)

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = np_to_torch(eps.observations)
        actions_flat = np_to_torch(eps.actions)
        rewards_flat = np_to_torch(eps.rewards)
        returns_flat = torch.cat(filter_valids(returns, valids))
        advs_flat = self._compute_advantage(rewards, valids, baselines)

        #safety-related things
        safety_rewards = np_to_torch(eps.padded_safety_rewards)
        safety_rewards_flat = np_to_torch(eps.safety_rewards)
        safety_returns = np_to_torch(np.stack([discount_cumsum(safety_reward, self.safety_discount)
                                                for safety_reward in eps.padded_safety_rewards]))

        #calculate safety baseline
        with torch.no_grad():
            safety_baselines = self._safety_baseline(obs)
        
        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            safety_rewards += self._policy_ent_coeff * policy_entropies

        safety_returns_flat = torch.cat(filter_valids(safety_returns, valids))
        safety_advs_flat = self._compute_safety_advantage(safety_rewards, valids, safety_baselines)

        #compute relevant metrics prior to training for logging
        with torch.no_grad():
            policy_loss_before = self._compute_loss_with_adv(obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_before = self._value_function.compute_loss(obs_flat, returns_flat)
            kl_before = self._compute_kl_constraint(obs)

        #train policy, value function, safety baseline
        self._train(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat, safety_rewards_flat, safety_returns_flat,
                    safety_advs_flat)

        #compute relevant metrics after training
        with torch.no_grad():
            policy_loss_after = self._compute_loss_with_adv(
                obs_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_after = self._value_function.compute_loss(obs_flat, returns_flat)
            kl_after = self._compute_kl_constraint(obs)
            policy_entropy = self._compute_policy_entropy(obs)

        #distributional RL
        #get mean and standard deviation of initial state in IMP
        vf_mean, vf_stddev = self.get_mean_std(self._value_function, self.initial_state)
        safety_mean, safety_stddev = self.get_mean_std(self._safety_baseline, self.initial_state)

        vf_quantile_probs = self.get_quantiles(self._value_function, self.initial_state)
        safety_quantile_probs = self.get_quantiles(self._safety_baseline, self.initial_state)
        
        #log interesting metrics
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
            tabular.record('/dLoss', vf_loss_before.item() - vf_loss_after.item())

            #value function distribution data
            if vf_mean is not None and vf_stddev is not None: 
                tabular.record('/MeanValue', vf_mean.item())
                tabular.record('/StdValue', vf_stddev.item())
            if vf_quantile_probs is not None: 
                for j in range(len(vf_quantile_probs)):
                    tabular.record(f'/QuantileProbability#{j}', vf_quantile_probs[j])

        with tabular.prefix('SafetyBaseline'):
            #safety distribution data
            if safety_mean is not None and safety_stddev is not None: 
                tabular.record('/MeanValue', safety_mean.item())
                tabular.record('/StdValue', safety_stddev.item())
            if safety_quantile_probs is not None: 
                for j in range(len(vf_quantile_probs)):
                    tabular.record(f'/QuantileProbability#{j}', safety_quantile_probs[j])

        self._old_policy.load_state_dict(self.policy.state_dict())

        undiscounted_returns = log_performance(itr,
                                               eps,
                                               discount=self._discount,
                                               safety_discount=self.safety_discount, 
                                               is_saute = self._is_saute)
        return np.mean(undiscounted_returns)
    

    def _train(self, obs, actions, rewards, returns, advs, 
               safety_rewards, safety_returns, safety_advs):
        r"""Train the policy and value function with minibatch.

        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            next_obs (torch.Tensor): Next observations from environment. 
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.

        """
        for dataset in self._policy_optimizer.get_minibatch(
                obs, actions, rewards, advs, safety_rewards, safety_advs):
            self._train_policy(*dataset)
        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._train_value_function(*dataset)
        for dataset in self._safety_optimizer.get_minibatch(obs, safety_returns):
            self.safety_constraint._train_safety_baseline(*dataset)


    def _train_policy(self, obs, actions, rewards, advantages, 
                      safety_rewards, safety_advantages):
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
        loss = self._compute_loss_with_adv(obs, actions, rewards, advantages)
        
        #if using conjugate constraint optimizer step has different args
        if self.safety_constrained_optimizer:
            #calculate objective gradients and normalise (if specified)
            loss_grad = self._get_grad(loss)
            if self.grad_norm:
                loss_grad = loss_grad/torch.norm(loss_grad) 

            #calculate safety_loss and grad
            #safety loss is in opposite direction as objective loss
            safety_loss = -self._compute_loss_with_adv(obs, actions, safety_rewards, safety_advantages)
            safety_loss_grad = self._get_grad(safety_loss)
            safety_loss_grad = safety_loss_grad/torch.norm(safety_loss_grad) 

            lin_leq_constraint = (lambda: -self._compute_loss_with_adv(obs, actions, safety_rewards, safety_advantages), 
                                  self.safety_step_size)
            
            quad_leq_constraint = (lambda: self._compute_kl_constraint(obs), self.step_size)

            self._policy_optimizer.step(
                f_loss= lambda: self._compute_loss_with_adv(obs, actions, rewards, advantages),
                lin_leq_constraint= lin_leq_constraint,                                           
                quad_leq_constraint= quad_leq_constraint, 
                loss_grad=loss_grad, 
                safety_loss_grad=safety_loss_grad)

            return loss, safety_loss
        else: 
            loss.backward()
            self._policy_optimizer.step(
                f_loss=lambda: self._compute_loss_with_adv(obs, actions, rewards,
                                                        advantages),
                f_constraint=lambda: self._compute_kl_constraint(obs))

            return loss

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [self.policy, self._value_function, 
                self._safety_baseline]
    
    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.
        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)
        else: #move state dictionaries of networks to device
            self._safety_optimizer.load_state_dict(
                state_dict_to(self._safety_optimizer.state_dict(), device))
            self._vf_optimizer.load_state_dict(
                state_dict_to(self._vf_optimizer.state_dict(), device))
            self._policy_optimizer.load_state_dict(
                state_dict_to(self._policy_optimizer.state_dict(), device))
    
