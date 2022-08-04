#misc
import os 
import random
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import csv
import warnings
import matplotlib.animation as ani

#jaisalab
from jaisalab.utils import get_time_stamp_as_string

class Plotter():
    """Plotting functionalities for evaluation of RL algorithms implemented 
    through the garage framework.
    
    Args: 
        - plot_latest (bool): Wether to plot the latest results found in the default
          data directory --> 'data/local/experiment'. If the default directory is specified 
          by the user, this must also be specified in the data_dir argument. 
        
        - fdir (str, list): Name of the experiment results directory within the data directory.
          If a list or tuple, all the experiments specified will be plotted. If None, plot_latest
          should be True. *NOTE*: It is assumed that fdirs (i.e. experiment functions) are named 
          in the format '{algorithm}_{environment}'. 

        - data_dir (str): Data directory. Default = 'data/local/experiment'.

        - dtype (str): Datatype of parsed results. Options=('np'), Default='np'.

        - savefig (bool): Wether to save plotted figures (By default these are saved to 'plots' 
          directory within the current working directory). 
    
    """
    def __init__(self, plot_latest=True, fdir=None, data_dir='data/local/experiment', 
                 dtype='np', savefig=True, **kwargs):

        self.data_dir = data_dir
        algorithm_names = ['cpo', 'trpo']

        if fdir is not None: 
            if isinstance(fdir, (list, tuple)): #multiple experiments inputted
                self.fdir = [data_dir+'/'+exp for exp in fdir]
                if len(fdir) > 4:
                    raise ValueError("Please don't input more than 4 experiments at once.")
                split_exp_names =[exp.split('_') for exp in fdir] 

                self._exp_labels = []
                for exp_name in split_exp_names:
                    if exp_name[1] in algorithm_names:
                        self._exp_labels.append('_'.join(exp_name[:2]))
                    else: 
                        self._exp_labels.append(exp_name[0])

                self._savefig_name = '_'.join(self._exp_labels)
            else: 
                #case where a single experiment is inputted
                self.fdir = data_dir+'/'+fdir
                self.dir_name = self.fdir.split('/')[-1]
                split_exp_name = self.dir_name.split('_')
                self._exp_label = split_exp_name[0]
                self._savefig_name = fdir
        else: 
            if plot_latest: #retrieve latest experiment in data directory
                list_of_data_dirs = [x[0] for x in os.walk(data_dir)]
                latest_data_dir = max(list_of_data_dirs, key=os.path.getctime)
                self.fdir = latest_data_dir
                self.dir_name = latest_data_dir.split('/')[-1]
                self._savefig_name = self.dir_name
                split_exp_name = self.dir_name.split('_')
                self._exp_label = split_exp_name[0]
            else: 
                raise TypeError("'NoneType' object is not an accesible data directory.")

        if dtype == 'np':
            self.data = self._collect_data_as_np()
        else: 
            raise NotImplementedError("Other datatypes have not yet been implemented.")

        self.savefig = savefig
        self.fig_kwargs = kwargs
        self._dtype = dtype
        self._colors = ['b', 'orange', 'g', 'r']
        self._plot_color = random.choice(self._colors)
        self._plot_flags = {1:'returns', 2:'kl', 
                            3:'constraint_vals', 4:'entropy',
                            5:'losses', 6:'costs', 7:'dist_progress', 
                            8:'final_dist', 9:'quantile_dist_progress'}

    def _get_data_dict(self, csvreader): 
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
                file_data_dict = self._get_data_dict(reader)
                data_dict[self.fdir[i]] = file_data_dict
        else: 
            data_dict = self._get_data_dict(csvreader)

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

    def _savefig(self,flag,animator=None):
        if self.savefig:
            try:
                if 'plots' not in os.listdir():
                    os.mkdir('plots/')
                
                if animator is not None: 
                    animator.save(f'plots/{self._savefig_name}_{self._plot_flags[flag]}.gif')
                else: 
                    plt.savefig(f'plots/{self._savefig_name}_{self._plot_flags[flag]}')

            except FileExistsError:
                pass
    
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
    
    def _get_data_arrays(self, y_column, std_column=None):
        if isinstance(self.fdir, (list, tuple)):
            y_array = [data[y_column] for data in self.data.values()]
            x_array = [np.arange(0, len(y)) for y in y_array]
            if std_column is not None:
                std_array = [data[std_column] for data in self.data.values()]
                return x_array, y_array, std_array
            else: 
                return x_array, y_array
        else: 
            y_array = self.data[y_column]
            x_array = np.arange(0, len(y_array))
            if std_column is not None:
                std_array = self.data[std_column]
                return x_array, y_array, std_array
            else: 
                return x_array, y_array
        
    def _get_distribution_data(self):
        if isinstance(self.fdir, (list, tuple)):
            if len(self.fdir) > 1:
                raise TypeError('Distribution progression can only be plotted for individual experiments.')

        col_names = ['MeanValue', 'StdValue']
        cols_ls, names_idxs = self._get_columns(col_names)
        mean_returns_array = self.data[cols_ls[names_idxs[0]]]
        std_returns_array = self.data[cols_ls[names_idxs[1]]]
        
        return mean_returns_array, std_returns_array
    
    def _get_quantiles_data(self, Vmin, Vmax):
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

    def _plot(self, x_array, y_array, ylabel, std=None, title=None):
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
        y_column = 'Evaluation/AverageReturn'
        std_column = 'Evaluation/StdReturn'
        ylabel = 'Average Return'
    
        x_array, y_array, std_array = self._get_data_arrays(y_column, std_column)

        self._plot(x_array, y_array, ylabel, std=std_array)
        self._savefig(flag=1)

    def plot_costs(self):
        """Plot progression of costs throughout epochs."""
        y_column = 'Evaluation/AverageSafetyReturn'
        std_column = 'Evaluation/StdSafetyReturn'
        ylabel = 'Average Costs'

        x_array, y_array, std_array = self._get_data_arrays(y_column, std_column)
        
        self._plot(x_array, y_array, ylabel, std=std_array)
        self._savefig(flag=6)

    def plot_constraint_vals(self):
        """Plot progression of constraint values throughout epochs."""
        y_column = 'Evaluation/AverageDiscountedSafetyReturn'
        ylabel = 'Constraint Value'

        x_array, y_array = self._get_data_arrays(y_column)

        self._plot(x_array, y_array, ylabel)
        self._savefig(flag=3)
    
    def _gaussian(self, mean, std):
        """Retrieve a normal distribution function from a mean and 
        standard deviation."""
        def normal(x):
            return (1/std*np.sqrt(2*np.pi))*np.exp(-(x - mean)**2 / (2*std**2))
        return normal

    def plot_gaussian_progression(self, eps=1.0, num_points=300, duration=5):
        """Plot the progression of normal distributions from learned means 
        and standard deviations throughout learning.
        
        Args: 
            eps (float, int): Factor proportional to standard deviation by which to 
                 extend domain of plots.
            num_points (int): Points used in creating gaussian distribution approximation.
            duration (int): Duration of animation.
        """
        mean_returns_array, std_returns_array = self._get_distribution_data()
        gaussians = [self._gaussian(mean, std) for mean, std in zip(mean_returns_array, std_returns_array)]
        min_x = min([mean - eps*std for mean, std in zip(mean_returns_array, std_returns_array)])
        max_x = max([mean + eps*std for mean, std in zip(mean_returns_array, std_returns_array)])

        x_array = np.linspace(min_x, max_x, num=num_points)
        y_arrays = [gaussian(x_array) for gaussian in gaussians]

        fig = plt.figure()
        def dist_progress(i):
            fig.clear()
            plt.ylim(0,1)
            p = plt.plot(x_array, y_arrays[i], c='r')
            plt.xlabel('Returns')
            plt.ylabel('Probability')
        interval = duration*1000 / len(y_arrays)
        animator = ani.FuncAnimation(fig, dist_progress, interval=interval, save_count=len(y_arrays))
        self._savefig(flag=7, animator=animator)

    def plot_final_distribution(self):
        """Plot final normal distribution after learning."""
        mean_returns_array, std_returns_array = self._get_distribution_data()
        last_mean, last_std = mean_returns_array[-1], std_returns_array[-1]
        gaussian = self._gaussian(last_mean, last_std)
        x_array = np.linspace(0, 600, num=1000)
        y_array = gaussian(x_array)

        fig = plt.figure()
        plt.plot(x_array, y_array, c='r')
        plt.xlabel('Return')
        plt.ylabel('Probability')
        self._savefig(flag=8)

    def plot_quantiles_progression(self, Vmin=-800, Vmax=800, interval=20, **kwargs):
        """Plot progression of quantile value distribution throughout learning."""
        quantile_probs, quantile_vals = self._get_quantiles_data(Vmin, Vmax)

        fig = plt.figure()
        def dist_progress(i):
            fig.clear()
            plt.ylim(0,1)
            p = plt.bar(quantile_vals, quantile_probs[i*interval], 
                        color='dimgrey', edgecolor='black', width=25)
            plt.title(f'State-Return Distribution for initial IMP state at epoch: {i*interval}')
            plt.xlabel('Value')
            plt.ylabel('Probability')

        time_interval = 5000 / len(quantile_probs)
        save_count = len(quantile_probs) // interval
        animator = ani.FuncAnimation(fig, dist_progress, interval=time_interval, save_count=save_count)
        self._savefig(flag=9, animator=animator)

    def plot_kl(self):
        pass

    def plot_losses(self):
        pass

    def plot_entropy(self):
        pass

    
    