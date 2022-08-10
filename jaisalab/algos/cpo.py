"""Constrained Policy Optimization using PyTorch with the garage framework."""
import torch

#jaisalab
from jaisalab.optimizers import ConjugateConstraintOptimizer
from jaisalab.safety_constraints import SoftInventoryConstraint
from jaisalab.algos import PolicyGradientSafe

#garage
from garage.torch.optimizers import OptimizerWrapper
from garage.torch._functions import zero_optim_grads


class CPO(PolicyGradientSafe):
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
        safety_constraint (jaisalab.safety_constraints.BaseConstraint): Environment safety constraint.
        safety_discount (float): Safety discount.
        safety_gae_lambda (float): Lambda used for generalized safety advantage
                                   estimation.        
        num_train_per_epoch (int): Number of train_once calls per epoch.
        step_size (float): Maximum KL-Divergence between old and new policies.
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
        grad_norm (bool): Wether to normalise the objective loss gradients. 
    """

    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 sampler,
                 policy_optimizer=None,
                 vf_optimizer=None,
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
                 grad_norm=False):

        #CPO uses ConjugateConstraintOptimizer
        if policy_optimizer is None:
            policy_optimizer = OptimizerWrapper(ConjugateConstraintOptimizer,
                                                policy)

        if vf_optimizer is None:
            vf_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=2.5e-4)),
                value_function,
                max_optimization_epochs=10,
                minibatch_size=64)

        if safety_constraint is None:
            safety_constraint = SoftInventoryConstraint()

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         sampler=sampler,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         safety_constraint=safety_constraint,
                         safety_discount=safety_discount,
                         safety_gae_lambda=safety_gae_lambda,
                         center_safety_vals=center_safety_vals,
                         num_train_per_epoch=num_train_per_epoch,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         grad_norm=grad_norm)
        
        self.max_quad_constraint = step_size 
        self.max_lin_constraint = self.safety_constraint.safety_step

    def _compute_objective(self, advantages, obs, actions, rewards):
        r"""TRPO Compute objective value. 

        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.

        """
        with torch.no_grad():
            old_ll = self._old_policy(obs)[0].log_prob(actions)

        new_ll = self.policy(obs)[0].log_prob(actions)
        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        return surrogate
    

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
        
        #calculate objective gradients and normalise (if specified)
        loss_grad = self._get_grad(loss)
        if self.grad_norm:
            loss_grad = loss_grad/torch.norm(loss_grad) 

        #calculate safety_loss and grad
        #safety loss is in opposite direction as objective loss
        safety_loss = -self._compute_loss_with_adv(obs, actions, safety_rewards, safety_advantages)
        safety_loss_grad = self._get_grad(safety_loss)
        safety_loss_grad = safety_loss_grad/torch.norm(safety_loss_grad) 

        #could input loss and arbitrary step size OR could input actual calculated costs
        #and use a more interpretable maximum value for its constraint
        lin_leq_constraint = (lambda: -self._compute_loss_with_adv(obs, actions, safety_rewards, safety_advantages), 
                                self.max_lin_constraint)         
        
        quad_leq_constraint = (lambda: self._compute_kl_constraint(obs), 
                                self.max_quad_constraint)

        self._policy_optimizer.step(
            f_loss= lambda: self._compute_loss_with_adv(obs, actions, rewards, advantages),
            lin_leq_constraint= lin_leq_constraint,                                           
            quad_leq_constraint= quad_leq_constraint, 
            loss_grad=loss_grad, 
            safety_loss_grad=safety_loss_grad)

        return loss, safety_loss