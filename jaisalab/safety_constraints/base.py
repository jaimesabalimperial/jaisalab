from xml.dom.minidom import Attr
from dowel import logger
import torch 
from garage.torch.optimizers import (
                                     OptimizerWrapper)
from garage.torch._functions import zero_optim_grads

class BaseConstraint(object):

    def __init__(self, max_value=1., baseline=None, 
                 baseline_optimizer=None, penalty=None, 
                 **kwargs):
        self.penalty = penalty
        self.max_value = max_value
        self.has_baseline = baseline is not None
        if self.has_baseline:
            self.baseline = baseline
        else: 
            raise AttributeError("Safety baseline cant be NoneType.")

        if baseline_optimizer is None:
            self.baseline_optimizer = OptimizerWrapper(
                    (torch.optim.Adam, dict(lr=2.5e-4)),
                    self.baseline,
                    max_optimization_epochs=10,
                    minibatch_size=64)
        else: 
            self.baseline_optimizer = OptimizerWrapper(
                    (baseline_optimizer, dict(lr=2.5e-4)),
                    self.baseline,
                    max_optimization_epochs=10,
                    minibatch_size=64)
        

    def evaluate(self, paths):
        raise NotImplementedError

    def _train_safety_baseline(self, obs, safety_returns):
        if self.has_baseline:
            # pylint: disable=protected-access
            zero_optim_grads(self.baseline_optimizer._optimizer)
            loss = self.baseline.compute_loss(obs, safety_returns)
            loss.backward()
            self.baseline_optimizer.step()
        else: 
            logger.log("Safety baseline has not been inputted...")

        return loss
            
    def get_safety_step(self):
        return self.max_value