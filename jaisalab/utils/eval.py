import numpy as np 
import os 

def order_experiments(data_dirs):
    """Checks that the data directories contain the same number of 
    experiments and that the experiments are consistent in naming. Also
    returns an ordered nested list containing the different replications 
    for each experiment. 
    """
    ordered_experiments = {}
    for dir in data_dirs:
        experiment_paths = [x[0] for x in os.walk(dir)][1:]
        experiment_names = [path.split('/')[-1] for path in experiment_paths]
        
def gather_replications(data_dirs):
    """Gathers the data from a variety of directories and averages the 
    data from equivalent experiments across the different ran seed values. 
    
    *NOTE*: Assumes that the data directories are structured as follows:

        - 1). Each directory contains the experiments for the same seed value 
        (e.g. TRPO, CPO, SAUTE_TRPO, and DCPO each ran a single time on the 
        Inventory management environment for a seed value of 1).
        
        - 2). The experiment names within the directories are consistent in 
        structure (i.e. are formatted as '{algo_name}_{env_name}_{seed_value}').
    
    Args: 

        data_dirs (tuple, list): List or tuple of strings specifying relative paths 
        to data directories. 
    """

    pass