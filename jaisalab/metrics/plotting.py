#misc
import os 
import glob
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import csv

#jaisalab
from jaisalab.utils import get_time_stamp_as_string

class Plotter():
    def __init__(self, plot_latest=True, fdir=None, data_dir=None, dtype='np', savefig=True, **kwargs):
        if data_dir is None:
            data_dir = 'data/local/experiment'

        self.data_dir = data_dir

        if plot_latest: 
            list_of_data_dirs = [x[0] for x in os.walk(data_dir)]
            latest_data_dir = max(list_of_data_dirs, key=os.path.getctime)
            self.fdir = latest_data_dir
            self.dir_name = latest_data_dir.split('/')[-1]
        else: 
            if fdir is not None: 
                self.fdir = fdir
            else: 
                raise TypeError("'NoneType' object is not an accesible data directory.")

        self.savefig = savefig
        self.fig_kwargs = kwargs

        if dtype == 'np':
            self.data = self.collect_data_as_np()
        elif dtype == 'pd':
            self.data = self.collect_data_as_pd()
        
        self.flags = {1:'returns', 2:'kl', 
                      3:'constraint_vals', 4:'entropy',
                      5:'losses', 6:'costs'}

    def _collect_data(self):
        file = open(f'{self.fdir}/progress.csv')
        csvreader = csv.reader(file) #csv reader
        #get metric names
        metrics = [] 
        metrics = next(csvreader)
        #extract data
        data = []
        for row in csvreader:
            data.append(row)
        #convert to NumPy array
        data = np.array(data)
        data = data.astype(np.float32)

        data_dict = {}
        for i, metric in enumerate(metrics):
            
            data_dict[metric] = data[:,i]

        return data_dict

    def collect_data_as_np(self):
        data_dict = self._collect_data()
        for metric, values in data_dict.items():
            data_dict[metric] = np.array(values)
        return data_dict

    def collect_data_as_pd(self):
        data_dict = self._collect_data()
        df = pd.DataFrame.from_dict(data_dict)
        return df

    def _savefig(self,flag):
        if self.savefig:
            try:
                if 'plots' not in os.listdir():
                    os.mkdir('plots/')
                plt.savefig(f'plots/{self.dir_name}_{self.flags[flag]}')
            except FileExistsError:
                pass
    
    def _multiple_inputs(self, x, y):
        if (all(isinstance(i, list) for i in x) 
            and all(isinstance(i, list) for i in y)):
            return True 
        return False
    
    def _plot(self, x, y, std=None, labels=None):
        if self._multiple_inputs(x,y):
            pass
        else: 
            fig = plt.figure()
            plt.grid()
            if std: 
                pass


    def plot_kl(self):
        pass
    def plot_losses(self):
        pass

    def plot_returns(self):
        avg_returns = self.data['Evaluation/AverageReturn']
        episodes = np.arange(0, len(avg_returns))
        std_returns = self.data['Evaluation/StdReturn']

        fig = plt.figure()
        plt.grid()
        plt.plot(episodes, avg_returns, color='b', label='Average Return')
        plt.fill_between(episodes, avg_returns-std_returns, avg_returns+std_returns,  color="b", alpha=0.4)
        plt.xlabel('Episode')
        plt.ylabel('Returns')
        plt.legend(loc='best')
        self._savefig(flag=1)

    def plot_entropy(self):
        pass

    def plot_costs(self):
        costs = self.data['Evaluation/AverageCosts']
        episodes = np.arange(0, len(costs))

        fig = plt.figure()
        plt.grid()
        plt.plot(episodes, costs, color='b', label='Average Costs')
        plt.xlabel('Episode')
        plt.ylabel('AvgCosts')
        plt.legend(loc='best')
        self._savefig(flag=6)

    def plot_constraint_vals(self):
        constraint_val = self.data['Evaluation/ConstraintValue']
        episodes = np.arange(0, len(constraint_val))

        fig = plt.figure()
        plt.grid()
        plt.plot(episodes, constraint_val, color='b', label='Constraint Value')
        plt.xlabel('Episode')
        plt.ylabel('Constraint Value')
        plt.legend(loc='best')
        self._savefig(flag=3)
    
    