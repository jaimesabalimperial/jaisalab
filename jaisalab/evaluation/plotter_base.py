#misc
import os 
import random
import numpy as np 
import matplotlib.pyplot as plt 
import csv

#jaisalab
from jaisalab.utils.eval import (gather_replications, get_data_dict, 
                                 order_experiments, _get_labels_from_dirs)

class BasePlotter():
    """Plotter base class."""
    def __init__(self, get_latest=True, fdir=None, data_dir='data/local/experiment', 
                 dtype='np', savefig=True, **kwargs):

        self.data_dir = data_dir
        self.algorithm_names = ['cpo', 'trpo']
        self._dtype = dtype

        #initialise plotter and gather data from directory/ies
        self._init_plotter(fdir, data_dir, get_latest) 

        self._savefig = savefig
        self.fig_kwargs = kwargs
        self._dtype = dtype
        self._colors = ['b', 'orange', 'g', 'r']
        self._plot_color = random.choice(self._colors)
        self._plot_flags = {1:'returns', 2:'kl', 
                            3:'constraint_vals', 4:'entropy',
                            5:'losses', 6:'costs', 7:'dist_progress', 
                            8:'final_dist', 9:'quantile_dist_progress'}
    
    def _init_plotter(self, fdir, data_dir, get_latest):
        """Initialise plotter attributes; includes gathering data from 
        specified directory or directories, and setting plotting arguments 
        depending on data collected."""
        #if one directory is specified then 
        if isinstance(data_dir, str):
            self.std_data = None
            if fdir is not None: 
                if isinstance(fdir, (list, tuple)): #multiple experiments inputted
                    self.fdir = [data_dir+'/'+exp for exp in fdir]
                    if len(fdir) > 4:
                        raise ValueError("Please don't input more than 4 experiments at once.")
                    
                    self._exp_labels = _get_labels_from_dirs(fdir, self.algorithm_names)
                    self._savefig_name = '_'.join(self._exp_labels)
                else: 
                    #case where a single experiment is inputted
                    self.fdir = data_dir+'/'+fdir
                    self.dir_name = self.fdir.split('/')[-1]
                    split_exp_name = self.dir_name.split('_')
                    self._exp_label = split_exp_name[0]
                    self._savefig_name = fdir
            else: 
                if get_latest: #retrieve latest experiment in data directory
                    experiment_paths = [x[0] for x in os.walk(data_dir)]
                    latest_experiment = max(experiment_paths, key=os.path.getctime)
                    self.fdir = latest_experiment
                    self.dir_name = latest_experiment.split('/')[-1]
                    self._savefig_name = self.dir_name
                    split_exp_name = self.dir_name.split('_')
                    self._exp_label = split_exp_name[0]
                else: 
                    raise TypeError("'NoneType' object is not an accesible data directory.")

            if self._dtype == 'np':
                self.data = self._collect_data_as_np()
            else: 
                raise NotImplementedError("Other datatypes have not yet been implemented.")

        #if multiple directories are specified then gather the data from the replications
        elif isinstance(data_dir, (tuple, list)):
            self.data, self.std_data = gather_replications(data_dir) #gather replications data
            ordered_experiments = order_experiments(data_dir)
            self.fdir = list(ordered_experiments.keys())
            self._exp_labels = _get_labels_from_dirs(self.fdir, self.algorithm_names)
            self._savefig_name = '_'.join(self._exp_labels)
        else: 
            raise TypeError('Specified data_dir must be a string or tuple/list of strings.')
        
    def _collect_data(self):
        if isinstance(self.fdir, (list, tuple)):
            if len(self.fdir) > 1:
                files = [open(f'{file}/progress.csv') for file in self.fdir]
                csvreader = [csv.reader(file) for file in files]#csv readers
            else: 
                file = open(f'{self.fdir[0]}/progress.csv')
                csvreader = csv.reader(file) #csv reader
        else: 
            file = open(f'{self.fdir}/progress.csv')
            csvreader = csv.reader(file) #csv reader

        if isinstance(csvreader, list):
            data_dict = {}
            for i, reader in enumerate(csvreader):
                file_data_dict = get_data_dict(reader)
                data_dict[self.fdir[i]] = file_data_dict
        else: 
            data_dict = get_data_dict(csvreader)

        return data_dict

    def _collect_data_as_np(self):
        data_dict = self._collect_data()
        if isinstance(self.fdir, (list, tuple)):
            if len(self.fdir) > 1:
                for exp in self.fdir:
                    for metric, values in data_dict[exp].items():
                        data_dict[exp][metric] = np.array(values)
            else: 
                for metric, values in data_dict.items():
                    data_dict[metric] = np.array(values)
        else:
            for metric, values in data_dict.items():
                data_dict[metric] = np.array(values)
        return data_dict

    def _get_columns(self, col_names):
        col_suffixes = []
        col_names_ls = []
        for col in self.data.keys():
            sep_col = col.split('/')
            if len(sep_col) > 1:
                col_suffix = sep_col[1]
                if col_suffix in col_names:
                    col_suffixes.append(col_suffix)
                    col_names_ls.append(col) 
        
        if len(col_names_ls) != len(col_names): 
            raise KeyError('Specified columns were not found.')
    
        names_idxs = [col_suffixes.index(name) for name in col_names]

        return col_names_ls, names_idxs
    
    def savefig(self, flag, animator=None):
        """Save a plotted figure (which could be animated)."""
        if self._savefig:
            try:
                if 'plots' not in os.listdir():
                    os.mkdir('plots/')
                
                if animator is not None: 
                    animator.save(f'plots/{self._savefig_name}_{self._plot_flags[flag]}.gif')
                else: 
                    plt.savefig(f'plots/{self._savefig_name}_{self._plot_flags[flag]}')

            except FileExistsError:
                pass
    
    
    def get_data_arrays(self, y_column, std_column=None):
        """Retrieve data arrays in different ways depending on data structure."""
        if isinstance(self.fdir, (list, tuple)):
            y_array = [data[y_column] for data in self.data.values()]
            x_array = [np.arange(0, len(y)) for y in y_array]

            if self.std_data is not None: 
                std_array = [data[y_column] for data in self.std_data.values()]
                return x_array, y_array, std_array
            else: 
                if std_column is not None:
                    std_array = [data[std_column] for data in self.data.values()]
                    return x_array, y_array, std_array
                else: 
                    return x_array, y_array
        else: 
            y_array = self.data[y_column]
            x_array = np.arange(0, len(y_array))

            if self.std_data is not None:
                std_array = self.std_data[y_column]
                return x_array, y_array, std_array
            else:
                if std_column is not None:
                    std_array = self.data[std_column]
                    return x_array, y_array, std_array
                else: 
                    return x_array, y_array
        
    def get_distribution_data(self):
        if isinstance(self.fdir, (list, tuple)):
            if len(self.fdir) > 1:
                raise TypeError('Distribution progression can only be plotted for individual experiments.')

        col_names = ['MeanValue', 'StdValue']
        cols_ls, names_idxs = self._get_columns(col_names)
        mean_returns_array = self.data[cols_ls[names_idxs[0]]]
        std_returns_array = self.data[cols_ls[names_idxs[1]]]
        
        return mean_returns_array, std_returns_array
    
    def get_quantiles_data(self, Vmin, Vmax):
        if isinstance(self.fdir, (list, tuple)):
            if len(self.fdir) > 1:
                raise TypeError('Distribution progression can only be plotted for individual experiments.')

        N = max([int(col.split('#')[-1]) for col in self.data.keys() if '#' in col]) + 1
        col_names = [f'QuantileProbability#{j}' for j in range(N)]

        cols_ls, names_idxs = self._get_columns(col_names)
        #retrieve quantile probabilities
        quantile_probs = []
        for j in range(N):
            quantile_probs.append(self.data[cols_ls[names_idxs[j]]])
        
        quantile_probs = np.array(quantile_probs).T #transpose array so that indices correspond to epochs

        delta_z = (Vmax-Vmin)/(N-1)
        quantile_vals = [Vmin + i*delta_z for i in range(N)]
        
        return quantile_probs, quantile_vals

    def plot(self, x_array, y_array, ylabel, std=None, title=None):
        """Custom plotting function that allows for multiple experiments to be 
        plotted on the same figure if specified in the fdir argument of the constructor
        method."""
        fig = plt.figure()
        plt.grid()
        if isinstance(self.fdir, (list, tuple)): #multiple experiments to plot
            min_episode_num = min([len(episodes) for episodes in x_array])
            for i, (x,y) in enumerate(zip(x_array, y_array)):
                plt.plot(x[:min_episode_num], y[:min_episode_num], 
                         color=self._colors[i], label=self._exp_labels[i])
                if std is not None: 
                    plt.fill_between(x[:min_episode_num], y[:min_episode_num]-std[i][:min_episode_num], 
                                     y[:min_episode_num]+std[i][:min_episode_num], color=self._colors[i], 
                                     alpha=0.4)
        else: 
            fig = plt.figure()
            plt.grid()
            plt.plot(x_array, y_array, color=self._plot_color, label=self._exp_label)
            if std is not None: 
                plt.fill_between(x_array, y_array-std, y_array+std, 
                                 color=self._plot_color, alpha=0.4)

        if title: 
            plt.title(title)
            
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.legend(loc='best')
    
    def plot_returns(self):
        """Plot progression of returns throughout epochs."""
        raise NotImplementedError

    def plot_costs(self):
        """Plot progression of costs throughout epochs."""
        raise NotImplementedError

    
