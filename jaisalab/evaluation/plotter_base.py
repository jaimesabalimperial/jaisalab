#misc
import os 
import random
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import csv
import inspect

#jaisalab
from jaisalab.utils.eval import (gather_replications, get_data_dict, 
                                 order_experiments, get_labels_from_dirs)

class BasePlotter():
    """Plotter base class."""
    def __init__(self, get_latest=True, fdir=None, data_dir='data/local/experiment', 
                 dtype='np', savefig=True, use_legend=True, **kwargs):

        self.data_dir = data_dir
        self.algorithm_names = ['cpo', 'trpo', 'ablation', 'dcpo']
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
                    
                    self._exp_labels = get_labels_from_dirs(fdir, self.algorithm_names)
                    self._savefig_name = '_'.join(self._exp_labels)
                else: 
                    #case where a single experiment is inputted
                    self.fdir = data_dir+'/'+fdir
                    self._exp_label = fdir.split('_')[0]
                    self._savefig_name = fdir
            else: 
                if get_latest: #retrieve latest experiment in data directory
                    experiment_paths = [x[0] for x in os.walk(data_dir)]
                    latest_experiment = max(experiment_paths, key=os.path.getctime)
                    self.fdir = latest_experiment
                    self.dir_name = latest_experiment.split('/')[-1]
                    self._savefig_name = self.dir_name
                    self._exp_label = self.dir_name.split('_')[0]
                else: 
                    raise TypeError("'NoneType' object is not an accesible data directory.")

            if self._dtype == 'np':
                self.data = self._collect_data_as_np()
            else: 
                raise NotImplementedError("Other datatypes have not yet been implemented.")

        #if multiple directories are specified then gather the data from the replications
        elif isinstance(data_dir, (tuple, list)):
            self.data, self.std_data = gather_replications(data_dir, fdir) #gather replications data
            ordered_experiments = order_experiments(data_dir, fdir)
            #for plotting purposes
            if fdir is not None: 
                self.fdir = fdir
                if isinstance(fdir, (list, tuple)): #multiple experiments inputted
                    self._exp_labels = get_labels_from_dirs(fdir, self.algorithm_names)
                    self._savefig_name = '_'.join(self._exp_labels)
                else: 
                    self._exp_label = fdir.split('_')[0]
                    self._savefig_name = fdir
            else: 
                self.fdir = list(ordered_experiments.keys())
                self._exp_labels = get_labels_from_dirs(self.fdir, self.algorithm_names)
                self._savefig_name = '_'.join(self._exp_labels)
        else: 
            raise TypeError('Specified data_dir must be a string or tuple/list of strings.')
        
    def _collect_data(self):
        if isinstance(self.fdir, (list, tuple)):
            files = [open(f'{file}/progress.csv') for file in self.fdir]
            csvreader = [csv.reader(file) for file in files]#csv readers
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
            for exp in self.fdir:
                for metric, values in data_dict[exp].items():
                    data_dict[exp][metric] = np.array(values)
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
        """Retrieve data arrays in different ways depending on data structure.
        
        Args: 
            y_column (str): Key of metric to retrieve from data dictionary. 
            std_column (str): Key of the standard deviation of the y_column 
                metric to plot. 

        Returns: 
            tuple(list, array): Tuple containing array of x, y, and std values. 
        """
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
            #data structured differently in case of replications
            if self.std_data is not None:
                y_array = list(self.data.values())[0][y_column] 
                x_array = np.arange(0, len(y_array))
                std_array = list(self.std_data.values())[0][y_column]
                return x_array, y_array, std_array
            else:
                y_array = self.data[y_column]
                x_array = np.arange(0, len(y_array))
                if std_column is not None:
                    std_array = self.data[std_column]
                    return x_array, y_array, std_array
                else: 
                    return x_array, y_array
        
    def get_distribution_data(self):
        """Retrieve data relevant to the distribution of values estimated
        throughout training (mean and standard deviation)."""
        if isinstance(self.fdir, (list, tuple)):
            if len(self.fdir) > 1:
                raise TypeError('Distribution progression can only be plotted for individual experiments.')

        col_names = ['MeanValue', 'StdValue']
        cols_ls, names_idxs = self._get_columns(col_names)
        mean_returns_array = self.data[cols_ls[names_idxs[0]]]
        std_returns_array = self.data[cols_ls[names_idxs[1]]]
        
        return mean_returns_array, std_returns_array
    
    def get_quantiles_data(self, Vmin, Vmax):
        """Retrieve data concerning quantile distribution (should be plotted 
        only for case where QRValueFunction is used for the value function/safety 
        baseline.
        
        Args: 
            Vmin (int, float): Minimum range of values in distribution (should 
                be consistent with arguments for QRValueFunction). 
            Vmax (int, float): Maximum range of values in distribution (should 
                be consistent with arguments for QRValueFunction). 
        
        Returns: 
            quantile_probs (list): List containing quantile probabilities at 
                every training iteration. 
            quantile_vals (list): List containing positions of quantile probability 
                bins. 
        """
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

    def plot(self, x_array, y_array, ylabel, std=None, title=None, use_legend=True,
             hline=None, **kwargs):
        """Custom plotting function that allows for multiple experiments to be 
        plotted on the same figure if specified in the fdir argument of the constructor
        method.

        Extra arguments may be plotted that relate to the figure (global fontsize may 
        also be inputted).

        Args: 
            x_array (list): List containing the domain of the y_array, which is by default
                a list of integers corresponding to the number of training iterations. 
            y_array (list): List containing data for the metric to be plotted for all 
                experiments specified. 
            ylabel (str): Label of y-axis. 
            std (list): List containing standard deviation of y_array at every training 
                iteration over a number of seeds specified by len(self.data_dirs); Default=None. 
            title (str): Title of figure (Default=None). 
            use_legend (bool): Wether to plot a legend in the figure. 
            hline (float): y-coordinate of a horizontal line to be plotted in the figure 
                (Default=None).
        """
        if 'fontsize' in kwargs.keys():
            plt.rcParams.update({'font.size': kwargs['fontsize']})
        if 'custom_labels' in kwargs.keys(): #allow for custom labels to be inputted
            labels = kwargs['custom_labels']
            if len(labels) != len(self._exp_labels):
                raise IndexError('Number of labels must match number of experiments.')
        else: 
            labels = self._exp_labels

        #handle different cases where single/multiple experiments are inputted
        fig_kwargs = {k:v for k,v in kwargs.items() if k in inspect.getargspec(plt.figure)[0]}
        fig = plt.figure(**fig_kwargs)
        plt.grid()
        if isinstance(self.fdir, (list, tuple)): #multiple experiments to plot
            min_episode_num = min([len(episodes) for episodes in x_array])
            for i, (x,y) in enumerate(zip(x_array, y_array)):
                plt.plot(x[:min_episode_num], y[:min_episode_num], 
                         color=self._colors[i], label=labels[i])
                if std is not None: 
                    plt.fill_between(x[:min_episode_num], y[:min_episode_num]-std[i][:min_episode_num], 
                                     y[:min_episode_num]+std[i][:min_episode_num], color=self._colors[i], 
                                     alpha=0.4)
        else: 
            plt.plot(x_array, y_array, color=self._plot_color, label=self._exp_label)
            if std is not None: 
                plt.fill_between(x_array, y_array-std, y_array+std, 
                                 color=self._plot_color, alpha=0.4)
        #horizontal line if specified
        if hline is not None: 
            hline_label = kwargs['hline_label']
            plt.axhline(hline, xmin=0, xmax=len(x_array),
                        linestyle='dashed', color='black', 
                        label=hline_label)

        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        
        if title: 
            plt.title(title)    
        if use_legend:
            plt.legend(loc='best')
        else: 
            if hline is not None: 
                hline_handle = Line2D([0], [0], linestyle='dashed', color='black')
                plt.legend([hline_handle], [hline_label], loc='best')
                
    
    def plot_returns(self):
        """Plot progression of returns throughout epochs."""
        raise NotImplementedError

    def plot_costs(self):
        """Plot progression of costs throughout epochs."""
        raise NotImplementedError

    
