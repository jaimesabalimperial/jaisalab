#misc
import torch
from dowel import tabular
import numpy as np

from garage import StepType
from garage.np import discount_cumsum

#jaisalab
from jaisalab.utils.misc import to_device

#garage
from garage import Trainer, wrap_experiment
from garage.experiment.deterministic import set_seed

def estimate_constraint_value(costs, masks, gamma, device):
    """Estimate the constraint value from the costs (i.e. safety rewards) 
    and the safety discount factor (gamma). Equivalent to the average discounted 
    safety returns.
    
    Args: 
        costs (torch.Tensor): Safety rewards observed. 
        masks (torch.Tensor): Boolean values specifying if timestep is 
                terminal. 
        gamma (float): Safety discount factor. 
        device (torch.device): Device to send calculated value to. 
    """
    costs, masks = to_device(torch.device('cpu'), costs, masks)
    constraint_value = torch.tensor(0)
    
    j = 1
    traj_num = 1
    for i in range(costs.size(0)):
        constraint_value = constraint_value + costs[i] * gamma**(j-1)
        
        if masks[i] == 0:
            j = 1 #reset
            traj_num = traj_num + 1
        else: 
            j = j+1
            
    constraint_value = constraint_value/traj_num
    constraint_value = to_device(device, constraint_value)
    return constraint_value[0]

@wrap_experiment
def resume_experiment(ctxt=None, 
                      experiment_name=None, 
                      n_epochs=None,
                      seed=1):
    """Function to resume experiment from its name in the data directory.
    
    Args: 
        experiment_name (str): Name of experiment in data directory (by default 
                'data/local/experiment/').
        n_epochs (int): Number of epochs to continue training for. 
        ctxt (garage.experiment.SnapshotConfig): Experiment context (default=None). 
        seed (int): Random seed to appy to experiment. 
    """
    if experiment_name is None: 
        raise ValueError('Experiment name must be specified.')
    if n_epochs is None: 
        raise ValueError('Number of epochs must be specified.')

    assert isinstance(experiment_name, str), 'Experiment name must be a string.'
    assert isinstance(n_epochs, int), 'n_epochs must be an integer. '
    
    #define snapshot_dir and set seed
    snapshot_dir= f'data/local/experiment/{experiment_name}'
    set_seed(seed)

    #resume experiment using trainer
    with Trainer(snapshot_config=ctxt) as trainer:
        trainer.restore(snapshot_dir)
        trainer.resume(n_epochs=n_epochs, batch_size=1024)

def gather_performance(batch, discount, safety_discount, is_saute):
    """Gather perfomance metrics from a batch of data in the form 
    of a SafeEpisodeBatch object. 
    
    Args: 
        batch (jaisalab.SafeEpisodeBatch): Batch of sampled data. 
        discount (float): Discount factor for returns. 
        safety_discount (float): Discount factor for safety returns (i.e. costs). 
        is_saute (bool): Boolean specifying if algorithm is sautéed. 

    Returns:
        dict[list or float]: Dictionary, with keys:
            * returns(list): Discounted returns observed for all episodes in batch.
            * undiscounted_returns(list): Undiscounted returns observed for all episodes in batch.
            * safety_returns(list): Discounted safety returns observed for all episodes in batch.
            * undiscounted_safety_returns(list): Undiscounted safety returns observed for all 
                episodes in batch.y of stacked,
            * termination(list): List indicating if episodes were terminated prematurely. 
            * success(list): List indicating if episodes weren't terminated prematurely.
            * average_return(float): Average returns observed throughout all episodes. 
            * average_safety_return(float): Average safety returns observed throughout all episodes. 
    """
    returns = []
    undiscounted_returns = []
    safety_returns = []
    undiscounted_safety_returns = []
    termination = []
    success = []
    for eps in batch.split():
        #account for saute algorithms
        if is_saute: 
            rewards = eps.env_infos['true_reward']
        else: 
            rewards = eps.rewards

        returns.append(discount_cumsum(rewards, discount))
        undiscounted_returns.append(sum(rewards))
        safety_returns.append(discount_cumsum(eps.safety_rewards, safety_discount))
        undiscounted_safety_returns.append(sum(eps.safety_rewards))
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))

    average_return = np.mean([rtn[0] for rtn in returns])
    average_safety_return = np.mean([rtn[0] for rtn in safety_returns])

    return dict(returns=returns, 
                undiscounted_returns=undiscounted_returns, 
                safety_returns=safety_returns, 
                undiscounted_safety_returns=undiscounted_safety_returns, 
                termination=termination, 
                success=success, 
                average_discounted_return=average_return, 
                average_discounted_safety_return=average_safety_return)

def log_performance(itr, batch, discount, safety_discount,
                    prefix='Evaluation', is_saute=False):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (SafeEpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    performance = gather_performance(batch, discount, safety_discount, is_saute)
    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
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

    return performance