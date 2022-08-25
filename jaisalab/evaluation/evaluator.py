#misc
import pandas as pd
import numpy as np
import os
from os import listdir
from dowel import logger, tabular
import dowel
import csv
from collections import defaultdict

#garage
from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed

#jaisalab
from jaisalab.sampler.sampler_safe import SamplerSafe
from jaisalab.algos.policy_gradient_safe import PolicyGradientSafe
from jaisalab.utils.agent import gather_performance
from jaisalab.utils.eval import get_data_dict, get_snapshot_dirs

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
                os.remove(eval_log_file)
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
        self.max_return = 330

        if isinstance(self.data['algo'], PolicyGradientSafe):
            self._is_saute = self.data['algo']._is_saute
        else:
            self._is_saute = False

        self._sampler = SamplerSafe(agents=self._policy,
                                envs=self._env, 
                                max_episode_length=self._max_episode_length, 
                                worker_args={'safety_constraint': self._safety_constraint})
    @property
    def eval_data(self):
        """Gather data logged during evaluation."""
        file = open(f'{self.snapshot_dir}/evaluation.csv')
        csvreader = csv.reader(file) #csv reader
        data_dict = get_data_dict(csvreader)

        for metric, values in data_dict.items():
            data_dict[metric] = np.array(values)
        
        return data_dict
    
    @property
    def returns(self):
        return self.eval_data['AverageDiscountedReturn']
    
    @property
    def costs(self):
        return self.eval_data['AverageDiscountedSafetyReturn']

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
        return epochs

    def get_mean_return(self):
        """Evaluate the mean return of the ran evaluation.
        
        Assumes the maximum constraint value is saved in an attribute 
        named 'max_lin_constraint'."""
        if not hasattr(self, 'data'):
            raise AttributeError('Evaluation has not been ran yet. Use rollout() method to run \
                an evaluation')

        test_return = self.eval_data['AverageDiscountedReturn']
        return np.mean(test_return) / self.max_return #dcpo normalisation 

    def get_mean_cost(self):
        """Evaluate the mean cost of the ran evaluation 
        as per the maximum constraint value specified in the algorithm.
        
        Assumes the maximum constraint value is saved in an attribute 
        named 'max_lin_constraint'."""
        if not hasattr(self, 'data'):
            raise AttributeError('Evaluation has not been ran yet. Use rollout() method to run \
                an evaluation')
        test_constraints = self.eval_data['AverageDiscountedSafetyReturn']
        return np.mean(test_constraints) / self.max_lin_constraint
    
    def num_violations(self):
        test_constraints = self.eval_data['AverageDiscountedSafetyReturn']
        num_violations = len(test_constraints[test_constraints > self.max_lin_constraint])
        return num_violations

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
        self.seed_data_dirs = [seed_dir + '/' + dir for dir in listdir(seed_dir)]
        self.experiment_tags =  ['cpo', 'trpo', 'ablation', 'dcpo']
        self.max_lin_constraint = 15 
        self.max_return = 330

        self._evaluators = defaultdict(list)
        for seed_dir in self.seed_data_dirs: 
            seed_tag = seed_dir[-1]
            snapshots = get_snapshot_dirs(seed_dir)
            exp_fdirs = [snapshot.split('/')[-1] for snapshot in snapshots]
            for exp_fdir, snapshot in zip(exp_fdirs, snapshots):
                eval_dict = {}
                eval_log_file = os.path.join(snapshot, 'evaluation.csv')
                eval_exists = os.path.exists(eval_log_file)
                eval_dict['evaluator'] = Evaluator(snapshot, override=override)
                eval_dict['rollout_necessary'] = not eval_exists
                eval_dict['experiment'] = exp_fdir
                self._evaluators[seed_tag].append(eval_dict)
    @property
    def raw_data(self):
        """Return raw returns and costs seed data."""
        #evaluate all experiments
        eval_dict = defaultdict(list)
        for seed_tag, eval_dicts in self._evaluators.items():
            for eval in eval_dicts:   
                evaluator = eval['evaluator']
                exp_name = eval['experiment']
                for tag in ['objective', 'safety']:
                    if tag == 'objective':
                        for ret in evaluator.returns:
                            eval_dict['experiment'].append(exp_name)
                            eval_dict['tag'].append(tag)
                            eval_dict['seed'].append(seed_tag)
                            eval_dict['metric'].append(ret / 330)
                    else:
                        for cost in evaluator.costs:
                            eval_dict['experiment'].append(exp_name)
                            eval_dict['tag'].append(tag)
                            eval_dict['seed'].append(seed_tag)
                            eval_dict['metric'].append(cost / 15)

        eval_df = pd.DataFrame(eval_dict)
        return eval_df
    
    def percentage_violations(self, experiment, max_cost=15.0):
        """Calculate the percentage of total evaluation points 
        that violate the safety constraint."""
        raw_data = self.raw_data 
        exps_to_remove = np.unique([exp for exp in raw_data.experiment if exp != experiment])
        data_copy = raw_data.copy()
        for exp in exps_to_remove:
            data_copy = data_copy.drop(data_copy[(data_copy['experiment'] == exp)].index)

        costs_data = data_copy[data_copy['tag'] == 'safety']
        violating_points = costs_data[costs_data['metric'] > max_cost]
        
        return len(violating_points['metric']) / len(costs_data['metric'])

    def rollout(self, n_epochs):
        """Rollout test data for all the trained policies in 
        the specified seed directory.       
        
        Args:  
            n_epochs (int): Number of batches of episodes to sample.
        """
        for seed_tag, eval_dicts in self._evaluators.items():
            for eval in eval_dicts:
                evaluator = eval['evaluator']
                exp_name = eval['experiment']
                print(f'Running evaluation for seed#{seed_tag} / {exp_name}')
                try:
                    evaluator.rollout(n_epochs)
                except FileExistsError:
                    print(f'Evaluation already exists in {evaluator.snapshot_dir}, moving on...')
                    continue
    
    def get_costs(self, experiment):
        """Returns the mean cost and its standard deviation across 
        the seeds in the specified seeds directory."""
        eval_dict = self.get_evaluation()
        experiment_results = eval_dict[eval_dict['experiment'] == experiment]
        experiment_costs =  experiment_results[experiment_results['tag'] == 'safety']['metric'] * self.max_lin_constraint 
        return experiment_costs

    def get_returns(self, experiment):
        """Returns the mean cost and its standard deviation across 
        the seeds in the specified seeds directory."""
        eval_dict = self.get_evaluation()
        experiment_results = eval_dict[eval_dict['experiment'] == experiment]
        experiment_returns =  experiment_results[experiment_results['tag'] == 'objective']['metric'] * self.max_return
        return experiment_returns
    
    def get_evaluation(self):
        """Returns a pd.DataFrame object with all of the evaluation results 
        for the experiments in the specified seeds directory. 
        
        Args: 
            eval_tag (str): Specified which metric to retrieve evaluation for. Must 
                be either 'task' (i.e. return) or 'safety' (i.e. cost).
        """
        #evaluate all experiments
        eval_dict = defaultdict(list)
        for seed_tag, eval_dicts in self._evaluators.items():
            for eval in eval_dicts:   
                evaluator = eval['evaluator']
                exp_name = eval['experiment']
                for tag in ['objective', 'safety']:
                    eval_dict['experiment'].append(exp_name)
                    eval_dict['tag'].append(tag)
                    eval_dict['seed'].append(seed_tag)
                    if tag == 'objective':
                        eval_dict['metric'].append(evaluator.get_mean_return())
                    else:
                        eval_dict['metric'].append(evaluator.get_mean_cost())

        eval_df = pd.DataFrame(eval_dict)
        return eval_df