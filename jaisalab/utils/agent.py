#misc
import torch

#jaisalab
from jaisalab.utils.misc import to_device

#garage
import garage
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
    