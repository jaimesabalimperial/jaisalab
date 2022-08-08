from tqdm import tqdm 
import numpy as np

#garage
from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed
from garage import StepType
from garage.np import discount_cumsum

#jaisalab
from jaisalab.sampler.sampler_safe import SamplerSafe
from jaisalab.algos.policy_gradient_safe import PolicyGradientSafe
from jaisalab.utils.agent import gather_performance

class Evaluator(object):
    """Class used to evaluate trained RL policies. Makes 
    use of garage's Snapshotter instance to extract the logged
    data by calling cloudpickle behind the scenes.
    
    Args: 
        snapshot_dir (str): Path to experiment snapshot directory. 
    """

    def __init__(self, snapshot_dir) -> None:
        self.data_dir = snapshot_dir
        self.snapshotter = Snapshotter()

        #examine data from snapshot of experiment
        self.data = self.snapshotter.load(snapshot_dir)
        self._policy = self.data['algo'].policy
        self._safety_constraint = self.data['algo'].safety_constraint
        self._env = self.data['env']
        self._seed = self.data['seed']
        self._max_episode_length = self._env.max_episode_length
        self._batch_size = self.data['train_args'].batch_size
        self._discount = self.data['algo']._discount
        self._safety_discount = self.data['algo'].safety_discount

        if isinstance(self.data['algo'], PolicyGradientSafe):
            self._is_saute = self.data['algo']._is_saute
        else:
            self._is_saute = False

        self._sampler = SamplerSafe(agents=self._policy,
                                envs=self._env, 
                                max_episode_length=self._max_episode_length, 
                                worker_args={'safety_constraint': self._safety_constraint})

    def _reset_evaluation(self):
        """Reset attributes for evaluation of paths."""
        self._returns = []
        self._undiscounted_returns = []
        self._safety_returns = []
        self._undiscounted_safety_returns = []
        self._termination = []
        self._success = []

    def rollout(self, n_epochs):
        """Obtain the paths sampled by the trained agent for 
        a given number of epochs.
        
        Args:  
            n_epochs (int): Number of batches of episodes to sample.
        
        Returns:
            paths (list): List containing paths in the form of 
                jaisalab._dtypes.SafeEpisodeBatch objects.
        """
        set_seed(self._seed)
        epochs = []
        for _ in tqdm(range(n_epochs), desc='Rolling out test batches...'):
            eps = self._sampler._obtain_samples(self._batch_size)
            epochs.append(eps)
        return epochs

    def evaluate_paths(self, epochs):
        """Evaluate the paths after rolling out n_epochs
         with a trained policy."""
        self._reset_evaluation()
        for batch in epochs:
            performance = gather_performance(batch, self._discount, 
                                             self._safety_discount, 
                                             self._is_saute)
            
            