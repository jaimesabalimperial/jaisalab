from tqdm import tqdm 

#garage
from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed

#jaisalab
from jaisalab.sampler.sampler_safe import SamplerSafe

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

        #with keys: ['seed', 'train_args', 'stats', 'env', 'algo', 
        #           'n_workers', 'worker_class', 'worker_args']
        self.data = self.snapshotter.load(snapshot_dir)
        self._policy = self.data['algo'].policy
        self._safety_constraint = self.data['algo'].safety_constraint
        self._env = self.data['env']
        self._seed = self.data['seed']
        self._max_episode_length = self._env.max_episode_length
        self._batch_size = self.data['train_args'].batch_size
        self._sampler = SamplerSafe(agents=self._policy,
                                envs=self._env, 
                                max_episode_length=self._max_episode_length, 
                                worker_args={'safety_constraint': self._safety_constraint})

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
        paths = []
        for _ in tqdm(range(n_epochs), desc='Rolling out test batches...'):
            eps = self._sampler._obtain_samples(self._batch_size)
            paths.append(eps)
        return paths

    def evaluate_paths(self, paths):
        pass