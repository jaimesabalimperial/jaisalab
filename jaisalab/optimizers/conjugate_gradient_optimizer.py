#garage
from garage.torch.optimizers import ConjugateGradientOptimizer
import warnings

class ConjugateGradOptimizer(ConjugateGradientOptimizer):
    """Taken from garage.torch.optimizers.ConjugateGradientOptimizer
    but changed __setstate__ method to support GPU usage."""

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