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
                 max_constraint_value,
                 cg_iters=10,
                 max_backtracks=15,
                 backtrack_ratio=0.8,
                 hvp_reg_coeff=1e-5,
                 accept_violation=False,
                 ):
        super().__init__(params, {})
        self._max_constraint_value = max_constraint_value
        self._cg_iters = cg_iters
        self._max_backtracks = max_backtracks
        self._backtrack_ratio = backtrack_ratio
        self._hvp_reg_coeff = hvp_reg_coeff
        self._accept_violation = accept_violation

    def step(self, f_loss, f_cost, f_constraint):  # pylint: disable=arguments-differ
        """Take an optimization step.

        Args:
            f_loss (callable): Function to compute the loss.
            f_constraint (callable): Function to compute the constraint value.

        """
        # Collect trainable parameters and gradients
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad.reshape(-1))
        flat_loss_grads = torch.cat(grads)

        # Build Hessian-vector-product function
        f_Ax = _build_hessian_vector_product(f_constraint, params,
                                             self._hvp_reg_coeff)

        # Compute step direction
        step_dir = _conjugate_gradient(f_Ax, flat_loss_grads, self._cg_iters)

        # Replace nan with 0.
        step_dir[step_dir.ne(step_dir)] = 0.

        # Compute step size
        step_size = np.sqrt(2.0 * self._max_constraint_value *
                            (1. /
                             (torch.dot(step_dir, f_Ax(step_dir)) + 1e-8)))

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
            for prev, cur in zip(prev_params, params):
                cur.data = prev.data
