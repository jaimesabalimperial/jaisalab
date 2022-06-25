"""Conjugate Constraint Optimizer.

Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
of the loss function.

Inspired by git@github.com:jachiam/cpo.git (specifically cpo/optimizers/conjugate_constraint_optimizer.py)
"""

import warnings

from dowel import logger
import numpy as np
import torch
from torch.optim import Optimizer

from garage.np import unflatten_tensors
from garage.torch.optimizers.conjugate_gradient_optimizer import _build_hessian_vector_product, _conjugate_gradient
from jaisalab.utils.torch import *
from jaisalab.utils.math import *


class ConjugateConstraintOptimizer(Optimizer):
    """Performs constrained optimization via backtracking line search.

    The search direction is computed using a conjugate gradient algorithm,
    which gives x = A^{-1}g, where A is a second order approximation of the
    constraint and g is the gradient of the loss function.

    Args:
        params (iterable): Iterable of parameters to optimize.
        max_constraint_value (float): Maximum constraint value.
        cg_iters (int): The number of CG iterations used to calculate A^-1 g
        max_backtracks (int): Max number of iterations for backtrack
            linesearch.
        backtrack_ratio (float): backtrack ratio for backtracking line search.
        hvp_reg_coeff (float): A small value so that A -> A + reg*I. It is
            used by Hessian Vector Product calculation.
        accept_violation (bool): whether to accept the descent step if it
            violates the line search condition after exhausting all
            backtracking budgets.

    """

    def __init__(self,
                 params,
                 max_kl,
                 cg_iters=10,
                 max_backtracks=25,
                 backtrack_ratio=0.1,
                 hvp_reg_coeff=1e-5,
                 accept_violation=False,
                 grad_norm=False):
        super().__init__(params, {})
        self._max_kl = max_kl
        self._cg_iters = cg_iters
        self._max_backtracks = max_backtracks
        self._backtrack_ratio = backtrack_ratio
        self._hvp_reg_coeff = hvp_reg_coeff
        self._accept_violation = accept_violation
        self._grad_norm = grad_norm

    def f_a_lambda(self, r, s, q, cc, lamda):
        a = ((r**2)/s - q)/(2*lamda)
        b = lamda*((cc**2)/s - self._max_kl)/2
        c = - (r*cc)/s
        return a+b+c
    
    def f_b_lambda(self, q, lamda):
        a = -(q/lamda + lamda*self._max_kl)/2
        return a   

    def _get_optimal_step_dir(self, cost_loss_grad, loss_grad, step_dir, 
                              cost_step_dir, constraint_value, d_k, f_Ax):
        """Solve optimisation problem using duality if algorithm takes a 
        step that produces a feasible iterate of the policy. However, sometimes, 
        it may take a bad step that produces an infeasible iterate of the policy 
        and when this happens an update that purely decreases the constraint value
        is used to recover from this bad step."""

        #define q, r, S
        q = -loss_grad.dot(step_dir) #g^T.H^-1.g
        r = loss_grad.dot(cost_step_dir) #g^T.H^-1.a
        S = -cost_loss_grad.dot(cost_step_dir) #a^T.H^-1.a 

        d_k = tensor(d_k).to(constraint_value.dtype).to(constraint_value.device)
        cc = constraint_value - d_k # c would be positive for most part of the training

        #find optimal lambda_a and lambda_b
        A = torch.sqrt((q - (r**2)/S)/(self._max_kl - (cc**2)/S))
        B = torch.sqrt(q/self._max_kl)
        if cc>0:
            opt_lam_a = torch.max(r/cc,A)
            opt_lam_b = torch.max(0*A,torch.min(B,r/cc))
        else: 
            opt_lam_b = torch.max(r/cc,B)
            opt_lam_a = torch.max(0*A,torch.min(A,r/cc))
        
        #find values of optimal lambdas 
        opt_f_a = self.f_a_lambda(r, S, q, cc, opt_lam_a)
        opt_f_b = self.f_b_lambda(q, opt_lam_b)
        
        if opt_f_a > opt_f_b:
            opt_lambda = opt_lam_a
        else:
            opt_lambda = opt_lam_b
            
        #find optimal nu
        nu = (opt_lambda*cc - r)/S
        if nu>0:
            opt_nu = nu 
        else:
            opt_nu = 0
            
        """ find optimal step direction """
        # check for feasibility
        if ((cc**2)/S - self._max_kl) > 0 and cc>0: #INFEASIBLE
            #opt_stepdir = -torch.sqrt(2*max_kl/s).unsqueeze(-1)*Fvp(cost_stepdir)
            opt_stepdir = torch.sqrt(2*self._max_kl/S)*f_Ax(cost_step_dir)
        else: #FEASIBLE
            #opt_grad = -(loss_grad + opt_nu*cost_loss_grad)/opt_lambda
            opt_stepdir = (step_dir - opt_nu*cost_step_dir)/opt_lambda
            #opt_stepdir = conjugate_gradients(Fvp, -opt_grad, 10)

        return opt_stepdir

    def step(self, f_loss, f_cost, f_constraint, loss_grad, cost_loss_grad,
             constraint_value, d_k):  # pylint: disable=arguments-differ
        """Take an optimization step.

        Args:
            f_loss (callable): Function to compute the loss.
            f_cost (callable): Function to compute cost loss.
            f_constraint (callable): Function to compute the constraint value.
            loss_grad: Gradient of objective loss. 
            cost_loss_grad : Gradient of cost loss.
            constraint_value : Value of constraint. 

        """
        # Collect trainable parameters and gradients
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)

        # Build Hessian-vector-product function
        f_Ax = _build_hessian_vector_product(f_constraint, params,
                                             self._hvp_reg_coeff)

        # Compute step direction
        step_dir = _conjugate_gradient(f_Ax, -loss_grad, self._cg_iters)
        cost_step_dir = _conjugate_gradient(f_Ax, -cost_loss_grad, self._cg_iters)

        # Calculate optimium step direction and replace nan with 0.
        opt_stepdir = self._get_optimal_step_dir(cost_loss_grad, loss_grad, step_dir, 
                                                 cost_step_dir, constraint_value, d_k, f_Ax)
        opt_stepdir[opt_stepdir.ne(opt_stepdir)] = 0.

        # Compute step size
        step_size = np.sqrt(2.0 * self._max_constraint_value *
                            (1. /(torch.dot(opt_stepdir, f_Ax(opt_stepdir)) + 1e-8)))

        if np.isnan(step_size):
            step_size = 1.

        descent_step = step_size * step_dir

        # Update parameters using backtracking line search
        self._backtracking_line_search(params, descent_step, f_loss,
                                       f_cost, f_constraint)

    @property
    def state(self):
        """dict: The hyper-parameters of the optimizer."""
        return {
            'max_constraint_value': self._max_constraint_value,
            'cg_iters': self._cg_iters,
            'max_backtracks': self._max_backtracks,
            'backtrack_ratio': self._backtrack_ratio,
            'hvp_reg_coeff': self._hvp_reg_coeff,
            'accept_violation': self._accept_violation,
            'grad_norm': self._grad_norm
        }

    @state.setter
    def state(self, state):
        # _max_constraint_value doesn't have a default value in __init__.
        # The rest of thsese should match those default values.
        # These values should only actually get used when unpickling a
        self._max_constraint_value = state.get('max_constraint_value', 0.01)
        self._cg_iters = state.get('cg_iters', 10)
        self._max_backtracks = state.get('max_backtracks', 15)
        self._backtrack_ratio = state.get('backtrack_ratio', 0.8)
        self._hvp_reg_coeff = state.get('hvp_reg_coeff', 1e-5)
        self._accept_violation = state.get('accept_violation', False)
        self._grad_norm = state.get('grad_norm', False)

    def __setstate__(self, state):
        """Restore the optimizer state.

        Args:
            state (dict): State dictionary.

        """
        if 'hvp_reg_coeff' not in state['state']:
            warnings.warn(
                'Resuming ConjugateGradientOptimizer with lost state. '
                'This behavior is fixed if pickling from garage>=2020.02.0.')
        self.defaults = state['defaults']
        # Set the fields manually so that the setter gets called.
        self.state = state['state']
        self.param_groups = state['param_groups']

    def _backtracking_line_search(self, params, descent_step, f_loss,
                                  f_cost, f_constraint):
        prev_params = [p.clone() for p in params]
        ratio_list = self._backtrack_ratio**np.arange(self._max_backtracks)
        loss_before = f_loss()
        cost_loss_before = f_cost()

        param_shapes = [p.shape or torch.Size([1]) for p in params]
        descent_step = unflatten_tensors(descent_step, param_shapes)
        assert len(descent_step) == len(params)

        for ratio in ratio_list:
            for step, prev_param, param in zip(descent_step, prev_params,
                                               params):
                step = ratio * step
                new_param = prev_param.data - step
                param.data = new_param.data

            loss = f_loss()
            cost_loss = f_cost()
            constraint_val = f_constraint()
            if (loss < loss_before and cost_loss < cost_loss_before
                and constraint_val <= self._max_constraint_value):
                break

        if ((torch.isnan(loss) or torch.isnan(constraint_val)
             or loss >= loss_before
             or constraint_val >= self._max_constraint_value)
             or torch.isnan(cost_loss)
             or cost_loss >= cost_loss_before
             and not self._accept_violation):

            logger.log('Line search condition violated. Rejecting the step!')

            if torch.isnan(cost_loss):
                logger.log('Violated because cost loss is NaN')      
            if cost_loss >= cost_loss_before:
                logger.log('Violated because cost loss not improving')                  
            if torch.isnan(loss):
                logger.log('Violated because loss is NaN')
            if torch.isnan(constraint_val):
                logger.log('Violated because constraint is NaN')
            if loss >= loss_before:
                logger.log('Violated because loss not improving')
            if constraint_val >= self._max_constraint_value:
                logger.log('Violated because constraint is violated')

            logger.log("Performing step without line search...")
            for step, prev, cur in zip(descent_step, prev_params, params):
                cur.data = prev.data + step

