#misc
from tqdm import tqdm 
import numpy as np
import os
from dowel import logger, tabular
import dowel
import csv
import shutil
from collections import defaultdict

#garage
from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed

#jaisalab
from jaisalab.sampler.sampler_safe import SamplerSafe
from jaisalab.algos.policy_gradient_safe import PolicyGradientSafe
from jaisalab.utils.agent import gather_performance
from jaisalab.utils.eval import get_data_dict, get_snapshot_dirs, get_labels_from_dirs

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
                self._override_error = False
            else: 
                self._override_error = True
        else: 
            self._override_error = False
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
        tabular.record('TrainingIteration', self._step_itr)
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
            epochs (list): List containing paths in the form of 
                jaisalab._dtypes.SafeEpisodeBatch objects.
        """
        if self._override_error: 
            raise FileExistsError(f'Evaluation already exists in {self.snapshot_dir}, please set override=True \
                        in constructor method f you wish to override the previous results.')

        logger.log(f'Rolling out evaluation for {self.snapshot_dir}...')
        set_seed(self._seed)
        epochs = []
        for i in range(n_epochs):
            eps = self._sampler._obtain_samples(self._batch_size)
            epochs.append(eps)
            performance = gather_performance(eps, self._discount, 
                                             self._safety_discount, 
                                             self._is_saute)
            #log diagnostics
            self._step_itr = i+1
            self.log_diagnostics(performance)
            logger.dump_all(i)
            tabular.clear()
        
        #remove output from logger after evaluation is complete
        logger.remove_all() 
        self.data = self.retrieve_evaluation()
        return epochs
    
    def mean_normalised_return(self):
        """Evaluate the mean normalised cost of the ran evaluation 
        as per the maximum constraint value specified in the algorithm.
        
        Assumes the maximum constraint value is saved in an attribute 
        named 'max_lin_constraint'."""
        if not hasattr(self, 'data'):
            raise AttributeError('Evaluation has not been ran yet. Use rollout() method to run \
                an evaluation')
        test_constraints = self.data['AverageDiscountedReturn']
        norm_avg_cost = np.mean(test_constraints) / self.max_lin_constraint
        return norm_avg_cost


    def mean_normalised_cost(self):
        """Evaluate the mean normalised cost of the ran evaluation 
        as per the maximum constraint value specified in the algorithm.
        
        Assumes the maximum constraint value is saved in an attribute 
        named 'max_lin_constraint'."""
        if not hasattr(self, 'data'):
            raise AttributeError('Evaluation has not been ran yet. Use rollout() method to run \
                an evaluation')
        test_constraints = self.data['AverageDiscountedSafetyReturn']
        norm_avg_cost = np.mean(test_constraints) / self.max_lin_constraint
        return norm_avg_cost

class SeedEvaluator():
    """Allows for the quick evaluation of multiple seeds by 
    creating an army of Evaluator objects.
    
    Args: 
        seed_dir (str): Directory containing the training data for 
            a variety of seeds. Data in seed_dir should be structured 
            as follows: 

                            +------------+
                +----------->data{seed#1}+------>...
                |           +------------+
                |
            +-------+       +------------+
            |seed_dir+------>data{seed#2}+------>...
            +-------+       +------------+
                |
                |
                +-----------> ...                     +------------+
                |                              +------>experiment#1|  
                |                              |      +------------+
                |                              |
                |           +------------+     |      +------------+
                +----------->data{seed#n}+-----+------>experiment#2| 
                            +------------+     |      +------------+
                                               |
                                               |      +------------+
                                               +------>experiment#n| 
                                                      +------------+

            *NOTE*: The names of the experiments inside the seed directories 
            should match. 
    
    """                                               

    def __init__(self, seed_dir, override=False) -> None:
        self.seed_dir = seed_dir
        self.seed_data_dirs = get_snapshot_dirs(seed_dir)
        self.experiment_tags =  ['cpo', 'trpo', 'ablation', 'dcpo']

        self._evaluators = defaultdict(list)
        self._override_warnings = defaultdict(list)
        self._rollout_necessary = defaultdict(list)
        for seed_dir in self.seed_data_dirs: 
            seed_tag = seed_dir[-1]
            snapshots = get_snapshot_dirs(seed_dir)
            exp_fdirs = [snapshot.split('/')[-1] for snapshot in snapshots]
            exp_names = get_labels_from_dirs(exp_fdirs, self.experiment_tags)
            eval_dict = {}
            for exp_name, snapshot in zip(exp_names, snapshots):
                eval_log_file = os.path.join(snapshot, 'evaluation.csv')
                eval_exists = os.path.exists(eval_log_file)
                eval_dict['evaluator'] = Evaluator(snapshot, override=override)
                eval_dict['rollout_necessary'] = not eval_exists
                eval_dict['experiment'] = exp_name
                self._evaluators[seed_tag].append(eval_dict)

    def rollout(self, n_epochs):
        """Rollout test data for all the trained policies in 
        the specified seed directory.       
        
        Args:  
            n_epochs (int): Number of batches of episodes to sample.
        """
        for evaluators in self._evaluators.values():
            for evaluator in evaluators:
                try:
                    evaluator.rollout(n_epochs)
                except FileExistsError:
                    logger.log(f'Evaluation already exists in {evaluator.snapshot_dir}, moving on...')
                    continue
    
    def get_cost_eval(self):
        """Get the normalised cost mean and standard deviation 
        over the different seeds for all the experiments in the 
        seed directories."""
        #evaluate all experiments
        seed_normalised_costs = defaultdict(dict)
        for seed_tag, evaluators in self._evaluators.items():
            for evaluator in evaluators:   
                exp_name = evaluator.snapshot_dir.split('/')[-1]
                seed_normalised_costs[seed_tag][exp_name] = evaluator.mean_normalised_cost()
        
        #transpose the data
        transposed_data = defaultdict(list)
        for seed_tag, data in seed_normalised_costs.items():
            for exp, cost in data.items():
                transposed_data[exp].append(cost)

        mean_normalised_cost = {}
        std_normalised_cost = {}
        for exp, seeds_data in transposed_data.items():
            average = np.mean(seeds_data, axis=0)
            std = np.std(seeds_data, axis=0)
            mean_normalised_cost[exp] = average
            std_normalised_cost[exp] = std
        
        return mean_normalised_cost, std_normalised_cost