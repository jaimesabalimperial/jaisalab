"""Constrained Policy Optimization using PyTorch with the garage framework."""
import torch
import numpy as np
#jaisalab
from jaisalab.algos.cpo import CPO
from jaisalab.value_functions import QRValueFunction
from jaisalab.safety_constraints import SoftInventoryConstraint, BaseConstraint

#garage
from garage.torch._functions import zero_optim_grads


class DCPO(CPO):
    """Distributional Constrained Policy Optimization (DCPO).

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
        tolerance (float): Probability tolerance for the safety baselines' estimate 
            of the probability of the average costs being greater than the maximum 
            allowed cost in CPO (max_lin_constraint); Default=0.05.
        beta (float): Cost reshaping coefficient; Default=0. 
        dist_penalty (bool): Boolean specifying if cost reshaping should be ablated. 
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
                 grad_norm=False, 
                 safety_margin=0.10, 
                 beta=2, 
                 dist_penalty=False): #ablation 
                
        
        if safety_constraint is None:
            #by default use a quantile regression baseline and soft constraints for IMP
            safety_baseline = QRValueFunction(env_spec=env_spec,
                                              Vmin=0, 
                                              Vmax=60., 
                                              N=102, 
                                              hidden_sizes=(64, 64),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

            self.safety_constraint = SoftInventoryConstraint(baseline=safety_baseline)
        else: 
            if isinstance(safety_constraint, BaseConstraint):
                self.safety_constraint = safety_constraint
            else: 
                raise TypeError("Safety constraint has to inherit from BaseConstraint.")

        if not isinstance(value_function, QRValueFunction):
            raise TypeError('The value function must be an instance of \
                jaisalab.value_functions.QRValueFunction in DCPO.')

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
                         step_size=step_size, 
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         grad_norm=grad_norm)
        
        assert safety_margin > 0 and safety_margin < 1, 'Safety margin must be between 0 and 1.'
        assert beta > 0, 'Beta must be positive.'
        
        self.dist_penalty = dist_penalty
        self.safety_margin = safety_margin
        self.beta = beta

        if not isinstance(self._safety_baseline, QRValueFunction):
            raise TypeError('The safety baseline must be an instance of \
                jaisalab.value_functions.QRValueFunction in DCPO.')

        z_dist = self._safety_baseline.V_range
        self.max_constraint_idx = (torch.abs(z_dist - self.max_lin_constraint)).argmin()
        
    def reshape_constraint(self):
        """Compute new constraint value $\tilde{J}_{C}$ and its target $d$ (i.e. maximum allowed 
        $\tilde{J}_{C}$)such
        that: 

        \begin{equation}

            \tilde{J}_{C} = J{C} * (1 + \rho) + \beta * (J_{C} - d)

            \tilde{d} = d * (\tilde{J}_{C} / J{C})
        
        \end{equation}

        where $\rho$ is the surplus probability that J_{C} > d as per the estimated quantile 
        distribution of costs (using QRValueFunction) and $\beta$ is a weight coefficient assigned 
        to the distance between J_{C} and its target d. 
        
        To ensure that dual problem constraint inequality holds (i.e. $\tilde{J}_{C} - \tilde{d} > 0 
        \iff J_{C} - d > 0$) we must condition the reshaping of $J_{C}$ and $d$ such that it is 
        only done if  J_{C} / d > \beta / (1 + \rho + \beta) (assuming that \rho > 0, \beta > 0). 
        """
        #use initial state prediction of quantiles to retrieve baseline of constraint value
        with torch.no_grad():
            mean_quantile_probs = self.get_quantiles(self._safety_baseline, self.initial_state)

        if self.constraint_value - self.max_lin_constraint > 0: #constraint violated 
            P = sum(mean_quantile_probs[self.max_constraint_idx:]) #P(J > d)
            k = self.beta * P
        else:
            P = torch.Tensor([-sum(mean_quantile_probs[:self.max_constraint_idx])]) #P(J < d)
            k = torch.clamp(self.beta * P, min=-self.safety_margin, max=None).item()

        J = self.constraint_value * (1 + k) #new constraint
        d = self.max_lin_constraint * (1 + k) #new target

        #compute surplus probability of obtaining J_{C} > d as per safety baseline
        #surplus_prob = sum(mean_quantile_probs[self.max_constraint_idx:]) - self.tolerance
        #surplus_prob = max(surplus_prob, 0)

        #if (self.constraint_value / self.max_lin_constraint) > (self.beta / (1 + surplus_prob + self.beta)): 
            #calculate difference between constraint value and limit
        #    delta =  self.constraint_value - self.max_lin_constraint        
        #    constraint = self.constraint_value * (1 + surplus_prob) + self.beta * delta
            
            #update moving target for constraint limit
        #    self.c = self.max_lin_constraint * (constraint / self.constraint_value)
        #else: #solve dual problem without reshaping
        #    self.c = self.max_lin_constraint
        #    return self.constraint_value

        return J, d

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

        #distributional penalty for dcpo
        if self.dist_penalty: 
            J, d = self.reshape_constraint()

        #define linear (safety) and quadratic (kl) constraints
        lin_leq_constraint = (J, d)         
        
        quad_leq_constraint = (lambda: self._compute_kl_constraint(obs), 
                                self.max_quad_constraint)

        self._policy_optimizer.step(
            f_loss= lambda: self._compute_loss_with_adv(obs, actions, rewards, advantages),
            f_safety= lambda: -self._compute_loss_with_adv(obs, actions, safety_rewards, safety_advantages),
            lin_leq_constraint= lin_leq_constraint,                                           
            quad_leq_constraint= quad_leq_constraint, 
            loss_grad=loss_grad, 
            safety_loss_grad=safety_loss_grad,
            rescale_factor=self.rescale_factor)

        return loss, safety_loss