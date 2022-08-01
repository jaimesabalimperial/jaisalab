from xml.dom.minidom import Attr
from dowel import logger
import torch 
import inspect 

#garage
from garage.torch.optimizers import OptimizerWrapper
from garage.torch._functions import zero_optim_grads

#jaisalab
from jaisalab.utils.misc import soft_update

class BaseConstraint(object):

    def __init__(self, max_value=1., baseline=None, 
                 baseline_optimizer=None, penalty=None, 
                 discount=1., **kwargs):
        self.penalty = penalty
        self.max_value = max_value
        self.has_baseline = baseline is not None
        self.discount = discount
        self.target_baseline = None

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
        """Abstract method that all safety constraints must have."""
        raise NotImplementedError

    def _train_safety_baseline(self, obs, next_obs, safety_returns,
                               safety_rewards, masks):
        """Train safety baseline."""
        if self.has_baseline:
            # pylint: disable=protected-access
            zero_optim_grads(self.baseline_optimizer._optimizer)
            args, varargs, varkw, defaults = inspect.getargspec(self.baseline.compute_loss)

            #differentiate between Q-learning vs policy-gradient optimization steps
            if len(args)>3:
                loss = self.baseline.compute_loss(obs, next_obs, safety_rewards,
                                                  masks, target_vf=self.target_baseline, 
                                                  gamma=self.discount)
            else: 
                loss = self.baseline.compute_loss(obs, safety_returns) 

            loss.backward()
            self.baseline_optimizer.step()

            if self.target_baseline is not None:
                soft_update(self.baseline, self.target_baseline)
        else: 
            logger.log("Safety baseline has not been inputted...")

        return loss

    @property          
    def safety_step(self):
        return self.max_value