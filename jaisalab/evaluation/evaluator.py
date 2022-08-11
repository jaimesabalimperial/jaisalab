#misc
from tqdm import tqdm 
import numpy as np
import os
from dowel import logger, tabular
import dowel
import csv
import shutil

#garage
from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed

#jaisalab
from jaisalab.sampler.sampler_safe import SamplerSafe
from jaisalab.algos.policy_gradient_safe import PolicyGradientSafe
from jaisalab.utils.agent import gather_performance
from jaisalab.utils.eval import get_data_dict

class Evaluator(object):
    """Class used to evaluate trained RL policies. Makes 
    use of garage's Snapshotter instance to extract the logged
    data by calling cloudpickle behind the scenes.
    
    Args: 
        snapshot_dir (str): Path to experiment snapshot directory. 
    """

    def __init__(self, snapshot_dir, override=False) -> None:
        #initialise evaluator and log dirs
        self.snapshot_dir = snapshot_dir

        eval_log_file = os.path.join(snapshot_dir, 'evaluation.csv')
        eval_exists = os.path.exists(eval_log_file)
        if eval_exists:
            if override==True:
                logger.add_output(dowel.CsvOutput(eval_log_file))
                logger.add_output(dowel.StdOutput())
        else: 
            logger.add_output(dowel.CsvOutput(eval_log_file))
            logger.add_output(dowel.StdOutput())

        self.snapshotter = Snapshotter()

        #remove empty directory that Snapshotter creates by default
        if len(os.listdir('data/local/experiment')) == 0:
            shutil.rmtree('data/')

        #examine data from snapshot of experiment
        self.data = self.snapshotter.load(snapshot_dir)
        self._policy = self.data['algo'].policy
        self._safety_constraint = self.data['algo'].safety_constraint
        self._env = self.data['env']
        self._seed = self.data['seed']
        self._max_episode_length = self.data['algo'].max_episode_length
        self._batch_size = self.data['train_args'].batch_size
        self._discount = self.data['algo']._discount
        self._safety_discount = self.data['algo'].safety_discount
        self.max_lin_constraint = self.data['algo'].safety_constraint.safety_step

        if isinstance(self.data['algo'], PolicyGradientSafe):
            self._is_saute = self.data['algo']._is_saute
        else:
            self._is_saute = False

        self._sampler = SamplerSafe(agents=self._policy,
                                envs=self._env, 
                                max_episode_length=self._max_episode_length, 
                                worker_args={'safety_constraint': self._safety_constraint})

    def retrieve_evaluation(self):
        """Gather data logged during evaluation."""
        file = open(f'{self.snapshot_dir}/evaluation.csv')
        csvreader = csv.reader(file) #csv reader
        data_dict = get_data_dict(csvreader)

        for metric, values in data_dict.items():
            data_dict[metric] = np.array(values)
        
        return data_dict

    def log_diagnostics(self, performance):
        """Log evaluation data to csv file."""
        tabular.record('NumEpisodes', len(performance['returns']))
        tabular.record('AverageDiscountedReturn', performance['average_discounted_return'])
        tabular.record('AverageReturn', np.mean(performance['undiscounted_returns']))
        tabular.record('StdReturn', np.std(performance['undiscounted_returns']))
        tabular.record('MaxReturn', np.max(performance['undiscounted_returns']))
        tabular.record('MinReturn', np.min(performance['undiscounted_returns']))
        tabular.record('AverageDiscountedSafetyReturn', performance['average_discounted_safety_return'])
        tabular.record('AverageSafetyReturn', np.mean(performance['undiscounted_safety_returns']))
        tabular.record('StdSafetyReturn', np.std(performance['undiscounted_safety_returns']))
        tabular.record('TerminationRate', np.mean(performance['termination']))
        if performance['success']:
            tabular.record('SuccessRate', np.mean(performance['success']))
        
        #log data onto csv file
        logger.log(tabular)

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
        for i in tqdm(range(n_epochs), desc='Rolling out test batches...'):
            eps = self._sampler._obtain_samples(self._batch_size)
            epochs.append(eps)
            performance = gather_performance(eps, self._discount, 
                                             self._safety_discount, 
                                             self._is_saute)
            #log diagnostics
            self.log_diagnostics(performance)
            logger.dump_all(i)
            tabular.clear()

        return epochs

    def mean_normalised_cost(self):
        """Evaluate the paths after rolling out n_epochs
         with a trained policy."""
        data = self.retrieve_evaluation()
        test_constraints = data['AverageDiscountedSafetyReturn']
        norm_avg_cost = np.mean(test_constraints) / self.max_lin_constraint
        return norm_avg_cost

