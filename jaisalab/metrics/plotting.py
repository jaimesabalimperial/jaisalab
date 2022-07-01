#misc
from multiprocessing.sharedctypes import Value
import os 
import random
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import csv
import warnings

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
          should be True.

        - data_dir (str): Data directory. Default = 'data/local/experiment'.

        - dtype (str): Datatype of parsed results. Options=('np'), Default='np'.

        - savefig (bool): Wether to save plotted figures (By default these are saved to 'plots' 
          directory within the current working directory). 
    
    """
    def __init__(self, plot_latest=True, fdir=None, data_dir='data/local/experiment', 
                 dtype='np', savefig=True, **kwargs):

        self.data_dir = data_dir

        if fdir is not None: 
            if isinstance(fdir, (list, tuple)):
                self.fdir = [data_dir+'/'+exp for exp in fdir]
                if len(fdir) > 4:
                    raise ValueError("Please don't input more than 4 experiments at once.")
                split_exp_names =[exp.split('_') for exp in fdir] 
                self._exp_labels = [exp_name[0] for exp_name in split_exp_names]
                self._savefig_name = '_'.join(self._exp_labels)
            else: 
                self.fdir = data_dir+'/'+fdir
                split_exp_name = self.dir_name.split('_')
                self._exp_label = split_exp_name[0]
                self._savefig_name = fdir
        else: 
            if plot_latest: 
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
                            5:'losses', 6:'costs'}

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
            files = [open(f'{file}/progress.csv') for file in self.fdir]
            csvreader = [csv.reader(file) for file in files]#csv readers
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
            for exp in self.fdir:
                for metric, values in data_dict[exp].items():
                    data_dict[exp][metric] = np.array(values)
        else:
            for metric, values in data_dict.items():
                data_dict[metric] = np.array(values)
        return data_dict

    def _savefig(self,flag):
        if self.savefig:
            try:
                if 'plots' not in os.listdir():
                    os.mkdir('plots/')
                plt.savefig(f'plots/{self._savefig_name}_{self._plot_flags[flag]}')
            except FileExistsError:
                pass
    
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
        
    
    def _plot(self, x_array, y_array, ylabel, 
              std=None, title=None):
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
        y_column = 'Evaluation/AverageReturn'
        std_column = 'Evaluation/StdReturn'
        ylabel = 'Average Return'
    
        x_array, y_array, std_array = self._get_data_arrays(y_column, std_column)

        self._plot(x_array, y_array, ylabel, std=std_array)
        self._savefig(flag=1)

    def plot_costs(self):
        y_column = 'Evaluation/AverageCosts'
        ylabel = 'Average Costs'

        x_array, y_array = self._get_data_arrays(y_column)
        
        self._plot(x_array, y_array, ylabel)
        self._savefig(flag=6)

    def plot_constraint_vals(self):
        y_column = 'Evaluation/ConstraintValue'
        ylabel = 'Constraint Value'

        x_array, y_array = self._get_data_arrays(y_column)

        self._plot(x_array, y_array, ylabel)
        self._savefig(flag=3)
    
    def plot_kl(self):
        pass
    def plot_losses(self):
        pass
    def plot_entropy(self):
        pass

    
    