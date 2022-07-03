"""Trust Region Policy Optimization."""
from logging import warning
import torch

#garage
from garage.torch.optimizers import (ConjugateGradientOptimizer,
                                     OptimizerWrapper)

#jaisalab
from jaisalab.algos.policy_gradient_safe import PolicyGradientSafe
from jaisalab.safety_constraints import InventoryConstraints



class SafetyTRPO(PolicyGradientSafe):
    """Trust Region Policy Optimization (TRPO) that logs information
    about the costs intrinsic to the safety constraints of the environment.

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
                 is_saute=False):

        if policy_optimizer is None:
            policy_optimizer = OptimizerWrapper(
                (ConjugateGradientOptimizer, dict(max_constraint_value=step_size)),
                policy)
            
        else: 
            if not isinstance(policy_optimizer, ConjugateGradientOptimizer):
                warning("Policy Optimizer for TRPO should be ConjugateGradientOptimizer.")
            
            policy_optimizer = OptimizerWrapper(policy_optimizer,
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
        self.env_spec = env_spec

        #TRPO uses ConjugateGradientOptimizer
        safety_constrained_optimizer=False

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         sampler=sampler,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         safety_constrained_optimizer=safety_constrained_optimizer,
                         safety_constraint=safety_constraint,
                         safety_discount=safety_discount,
                         safety_gae_lambda=safety_gae_lambda,
                         center_safety_vals=center_safety_vals,
                         num_train_per_epoch=num_train_per_epoch,
                         step_size=step_size,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         is_saute=is_saute)
    

