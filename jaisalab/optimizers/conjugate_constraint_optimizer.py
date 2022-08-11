"""Conjugate Constraint Optimizer.

Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
of the loss function.

Adjustment to git@github.com:jachiam/cpo.git (specifically cpo/optimizers/conjugate_constraint_optimizer.py)
to fit garage framework and PyTorch. 

Author: Jaime Sabal Bermúdez
"""

import warnings

from dowel import logger, tabular
import numpy as np
import torch
from torch.optim import Optimizer

from garage.np import unflatten_tensors
from garage.torch.optimizers.conjugate_gradient_optimizer import _build_hessian_vector_product, _conjugate_gradient
from jaisalab.value_functions.modules import *
from jaisalab.utils.math import *


class ConjugateConstraintOptimizer(Optimizer):
    """Performs constrained optimization via backtracking line search.
    The search direction is computed using a conjugate gradient algorithm,
    which gives x = A^{-1}g, where A is a second order approximation of the
    constraint and g is the gradient of the loss function.

    Partially taken from https://github.com/jachiam/cpo.

    Args:
        params (iterable): Iterable of parameters to optimize.
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
                 cg_iters=10,
                 max_backtracks=25,
                 backtrack_ratio=0.1,
                 hvp_reg_coeff=1e-5,
                 accept_violation=False):
        super().__init__(params, {})
        self._cg_iters = cg_iters
        self._max_backtracks = max_backtracks
        self._backtrack_ratio = backtrack_ratio
        self._hvp_reg_coeff = hvp_reg_coeff
        self._accept_violation = accept_violation

    def _get_optimal_step_dir(self, f_Ax, params, step_dir, 
                              safety_loss_grad, lin_constraint):
        """Solve optimisation problem using duality if algorithm takes a 
        step that produces a feasible iterate of the policy. However, sometimes, 
        it may take a bad step that produces an infeasible iterate of the policy 
        and when this happens an update that purely decreases the constraint value
        is used to recover from this bad step."""

        #following https://github.com/jachiam/cpo/blob/master/optimizers/conjugate_constraint_optimizer.py
        approx_g = f_Ax(step_dir)
        q = step_dir.dot(approx_g) # approx = g^T H^{-1} g
        delta = 2 * self._max_quad_constraint_val

        eps = 1e-8
        c = lin_constraint - self._max_lin_constraint_val #should be > 0 if constraint 

        #need to adjust by rescale factor (avg trajectory length) since true constraint is 
        #an average over trajectories, not state-action pairs, so true_constraint = T*surrogate_objective
        #where T is the average trajectory length
        c /= (self.rescale_factor + eps) 

        if c > 0: 
            logger.log("warning! safety constraint is already violated")
        else: 
            self.last_safe_point = [p.clone() for p in params]
    
        #solve for dual variables lambda and nu
        if safety_loss_grad.dot(safety_loss_grad) <= eps:
            logger.log("Safety gradient is zero --> linear constraint not present (ignore implementation)")
            # if safety gradient is zero, linear constraint is not present;
            # ignore its implementation.
            lam = torch.sqrt(q / delta)
            nu = 0
            w = 0
            r,s,A,B = 0,0,0,0
            optim_case = 4
        else:
            norm_b = np.sqrt(safety_loss_grad.dot(safety_loss_grad))
            unit_b = safety_loss_grad / norm_b
            w = norm_b * _conjugate_gradient(f_Ax, unit_b, cg_iters=self._cg_iters)

            r = w.dot(approx_g) # approx = b^T H^{-1} g
            s = w.dot(f_Ax(w))    # approx = b^T H^{-1} b

            # figure out lambda coeff (lagrange multiplier for trust region)
            # and nu coeff (lagrange multiplier for linear constraint)
            A = q - r**2 / s                # this should always be positive by Cauchy-Schwarz
            B = delta - c**2 / s            # this one says whether or not the closest point on the plane is feasible

            # if (B < 0), that means the trust region plane doesn't intersect the safety boundary
            if c <0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                logger.log('entire TR is feasible!')
                optim_case = 3
            elif c < 0 and B > 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                logger.log('most of TR is feasible!')
                optim_case = 2
            elif c > 0 and B > 0:
                # x = 0 is infeasible (bad! unsafe!) and safety boundary intersects
                # ==> part of trust region is feasible
                # ==> this is 'recovery mode'
                logger.log('recovery mode: part of TR is feasible!')
                optim_case = 1
            else:
                # x = 0 infeasible (bad! unsafe!) and safety boundary doesn't intersect
                # ==> whole trust region infeasible
                # ==> optimization problem infeasible!!!
                logger.log("optimisation problem infeasible!")
                optim_case = 0

            # default dual vars, which assume safety constraint inactive
            # (this corresponds to either optim_case == 3,
            #  or optim_case == 2 under certain conditions)
            lam = torch.sqrt(q / delta)
            nu  = 0

            if optim_case == 2 or optim_case == 1:
                # dual function is piecewise continuous
                # on region (a):
                #
                #   L(lam) = -1/2 (A / lam + B * lam) - r * c / s
                # 
                # on region (b):
                #
                #   L(lam) = -1/2 (q / lam + delta * lam)
                # 

                lam_mid = r / c
                L_mid = - 0.5 * (q / lam_mid + lam_mid * delta)

                lam_a = torch.sqrt(A / (B + eps))
                L_a = -torch.sqrt(A*B) - r*c / (s + eps)                 
                # note that for optim_case == 1 or 2, B > 0, so this calculation should never be an issue

                lam_b = torch.sqrt(q / delta)
                L_b = -torch.sqrt(q * delta)

                #those lam's are solns to the pieces of piecewise continuous dual function.
                #the domains of the pieces depend on whether or not c < 0 (x=0 feasible),
                #and so projection back on to those domains is determined appropriately.
                if lam_mid > 0:
                    if c < 0:
                        # here, domain of (a) is [0, lam_mid)
                        # and domain of (b) is (lam_mid, infty)
                        if lam_a > lam_mid:
                            lam_a = lam_mid
                            L_a   = L_mid
                        if lam_b < lam_mid:
                            lam_b = lam_mid
                            L_b   = L_mid
                    else:
                        # here, domain of (a) is (lam_mid, infty)
                        # and domain of (b) is [0, lam_mid)
                        if lam_a < lam_mid:
                            lam_a = lam_mid
                            L_a   = L_mid
                        if lam_b > lam_mid:
                            lam_b = lam_mid
                            L_b   = L_mid

                    if L_a >= L_b:
                        lam = lam_a
                    else:
                        lam = lam_b

                else:
                    if c < 0:
                        lam = lam_b
                    else:
                        lam = lam_a

                nu = max(0, lam * c - r) / (s + eps)

        if nu == 0:
            logger.log("safety constraint is not active!")
            
        
        if optim_case > 0:
            flat_descent_step = (1. / (lam + eps) ) * (step_dir + nu * w )
        else:
            # current default behavior for attempting infeasible recovery:
            # take a step on natural safety gradient
            flat_descent_step = torch.sqrt(delta / (s + eps)) * w

        return flat_descent_step

    def step(self, f_loss, f_safety, lin_leq_constraint, 
             quad_leq_constraint, loss_grad, safety_loss_grad, 
            rescale_factor):  # pylint: disable=arguments-differ
        """Take an optimization step.
        Args:
            f_loss (callable): Function to compute the objective loss.
            f_safety (callable): Function to compute safety loss. 
            lin_leq_constraint (callable): Function to compute safety loss.
            quad_leq_constraint (callable): Function to compute the constraint value (KL-Divergence).
            loss_grad: Gradient of objective loss. 
            safety_loss_grad : Gradient of cost loss.
            constraint_value : Value of constraint. 
        """
        # Collect trainable parameters and gradients
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)

        constraint_term_1, constraint_value_1 = quad_leq_constraint
        constraint_term_2, constraint_value_2 = lin_leq_constraint
        
        self._max_quad_constraint_val = constraint_value_1
        self._max_lin_constraint_val = constraint_value_2
        self.rescale_factor = rescale_factor

        # Build Hessian-vector-product function
        f_Ax = _build_hessian_vector_product(func=constraint_term_1, params=params,
                                             reg_coeff=self._hvp_reg_coeff)

        # Compute step direction
        step_dir = _conjugate_gradient(f_Ax, loss_grad, self._cg_iters)

        # Calculate optimium step direction and replace nan with 0.
        flat_descent_step = self._get_optimal_step_dir(f_Ax, params, step_dir, 
                                                       safety_loss_grad, constraint_term_2)

        # Update parameters using backtracking line search
        self._backtracking_line_search(params, flat_descent_step, f_loss=f_loss,
                                       f_safety=f_safety, f_constraint=constraint_term_1)

    @property
    def state(self):
        """dict: The hyper-parameters of the optimizer."""
        return {
            'max_constraint_value': self._max_constraint_value,
            'cg_iters': self._cg_iters,
            'max_backtracks': self._max_backtracks,
            'backtrack_ratio': self._backtrack_ratio,
            'hvp_reg_coeff': self._hvp_reg_coeff,
            'accept_violation': self._accept_violation
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

    def __setstate__(self, state):
        """Restore the optimizer state.
        Args:
            state (dict): State dictionary.
        """
        if 'hvp_reg_coeff' not in state['state']:
            warnings.warn(
                'Resuming ConjugateGradientOptimizer with lost state. '
                'This behavior is fixed if pickling from garage>=2020.02.0.')
        #self.defaults = state['defaults']
        # Set the fields manually so that the setter gets called.
        self.state = state['state']
        self.param_groups = state['param_groups']

    def _backtracking_line_search(self, params, descent_step, f_loss,
                                  f_safety, f_constraint):
        prev_params = [p.clone() for p in params]
        ratio_list = self._backtrack_ratio**np.arange(self._max_backtracks)
        loss_before = f_loss()
        cost_loss_before = f_safety()

        param_shapes = [p.shape or torch.Size([1]) for p in params]

        #consider case where descent step is a torch.Tensor
        if isinstance(descent_step, torch.Tensor):
            descent_step = descent_step.detach().numpy()

        descent_step = unflatten_tensors(descent_step, param_shapes)
        assert len(descent_step) == len(params)

        for ratio in ratio_list:
            for step, prev_param, param in zip(descent_step, prev_params,
                                               params):
                step = ratio * step
                new_param = prev_param.data - step
                param.data = new_param.data

            loss = f_loss()
            cost_loss = f_safety()
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
                logger.log('Violated because safety loss is NaN')      
            if cost_loss >= cost_loss_before:
                logger.log('Violated because safety loss not improving')                  
            if torch.isnan(loss):
                logger.log('Violated because loss is NaN')
            if torch.isnan(constraint_val):
                logger.log('Violated because constraint is NaN')
            if loss >= loss_before:
                logger.log('Violated because loss not improving')
            if constraint_val >= self._max_constraint_value:
                logger.log('Violated because constraint is violated')

            #revert to previous params
            for step, prev, cur in zip(descent_step, prev_params, params):
                cur.data = prev.data 