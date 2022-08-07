import numpy as np 
import os 
from collections import defaultdict
import csv
import warnings

def order_experiments(data_dirs, fdir=None):
    """Checks that the data directories contain the same number of 
    experiments and that the experiments are consistent in naming. Also
    returns an ordered nested list containing the different replications 
    for each experiment. 

    Args: 
        data_dirs (tuple, list): List or tuple of strings specifying relative paths 
            to data directories. 

        fdir (tuple, list): experiments to gather from the data directories (Default=None 
            which gathers all of the experiments in the specified data_dirs).
    """
    assert fdir is None or isinstance(fdir, (tuple, list, str)), 'fdir must be a tuple or a list.'
    ordered_experiments = defaultdict(list)

    for dir in data_dirs:
        experiment_paths = [x[0] for x in os.walk(dir)][1:]
        experiment_names = [path.split('/')[-1] for path in experiment_paths]

        for name, path in zip(experiment_names, experiment_paths):
            if fdir is not None: 
                if name in fdir:
                    ordered_experiments[name].append(path)
            else:
                ordered_experiments[name].append(path)
    
    #check that all of the experiments have the same number of replications
    num_replications = list(set([len(exp_replications) for exp_replications in ordered_experiments.values()]))

    if len(num_replications) > 1: 
        raise ReplicationsError("Number of replications across different experiments doesn't match.")
    elif num_replications[0] != len(data_dirs):
        raise ReplicationsError(f"Number of replications ({num_replications[0]}) \
                                doesn't match number of directories ({len(data_dirs)}).")
    return ordered_experiments

        
def gather_replications(data_dirs, fdir=None):
    """Gathers the data from a variety of directories and averages the 
    data from equivalent experiments across the different ran seed values. 
    
    *NOTE*: Assumes that the data directories are structured as follows:

        - 1). Each directory contains the experiments for the same seed value 
        (e.g. TRPO, CPO, SAUTE_TRPO, and DCPO each ran a single time on the 
        Inventory management environment for a seed value of 1).
        
        - 2). The experiment names within the directories are consistent in 
        structure (i.e. are formatted as '{algo_name}_{env_name}') and the same 
        for different replications. 
    
    Args: 
        data_dirs (tuple, list): List or tuple of strings specifying relative paths 
        to data directories. 

        fdir (tuple, list): experiments to gather from the data directories. 
    """
    ordered_experiments = order_experiments(data_dirs, fdir)

    data = {}
    for exp, replication_paths in ordered_experiments.items():
        files = [open(f'{file}/progress.csv') for file in replication_paths]
        csvreaders = [csv.reader(file) for file in files]#csv readers
        rep_data = []
        #get replications data
        for reader in csvreaders:
            file_data_dict = get_data_dict(reader)
            rep_data.append(file_data_dict)
        
        transposed_data = _transpose_data(rep_data)

        #store data in form {'exp1_name': {'metric1': np.array(metric1_reps_data}, ..., 'metricN': ...}, ...}
        data[exp] = transposed_data
    
    #retrieve average of the logged metrics across replications
    average_data = defaultdict(dict)
    std_data = defaultdict(dict)

    for exp, data_dict in data.items():
        for metric, data in data_dict.items():
            average = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            average_data[exp][metric] = average 
            std_data[exp][metric] = std 
    
    return average_data, std_data

def _get_labels_from_dirs(dirs, algorithm_names):
    split_exp_names =[exp.split('_') for exp in dirs] 

    exp_labels = []
    for exp_name in split_exp_names:
        if exp_name[1] in algorithm_names:
            exp_labels.append('_'.join(exp_name[:2]))
        else: 
            exp_labels.append(exp_name[0])
    return exp_labels

def _transpose_data(replications):
    """Transpose data in the following way: 
    
    From: [{'metric1': rep1_metric1_data, ..., 'metricN': rep1_metricN_data}, ...,
           {'metric1': repN_metric1_data, ..., 'metricN': repN_metricN_data}]
           
    To: {'metric1': np.array([rep1_metric1_data, rep2_metric1_data, ..., repN_metric1_data]), ..., 
         'metricN': np.array([rep1_metricN_data, rep2_metricN_data, ..., repN_metricN_data])}
    """
    transposed_data = defaultdict(list)
    
    for rep_data in replications:
        for metric, data in rep_data.items():
            transposed_data[metric].append(data)

    #convert metrics data into NumPy arrays
    for metric, reps_data in transposed_data.items():
        transposed_data[metric] = np.array(reps_data)
    
    return transposed_data

def get_data_dict(csvreader): 
    """Retrieve dictionary with data from a csvreader object."""
    #get metric names
    metrics = [] 
    metrics = next(csvreader)

    #extract data
    data = []
    for row in csvreader:
        data.append(row)

    #convert to NumPy array
    data = np.array(data)

    bad_metrics = None
    try:
        data = data.astype(np.float32)
    except ValueError:
        #a column in the dataset cant be converted to desired datatype
        bad_metrics = []
        old_data = data.copy()
        data = np.zeros(np.shape(old_data), dtype=np.float32)
        for col in range(len(data[0])):
            if '' in old_data[:,col]:
                bad_metrics.append(metrics[col])
                warnings.warn(message=f'Couldnt convert {metrics[col]} array to type np.float32.')
                continue
            else: 
                data[:,col] = old_data[:,col].astype(np.float32)

    data_dict = {}
    for i, metric in enumerate(metrics):
        if bad_metrics is not None and metric in bad_metrics:
            continue
        data_dict[metric] = data[:,i]
    
    return data_dict

class ReplicationsError(Exception):
    """Exception to be raised when the number of replications for 
    the different ran experiments don't match."""