import os 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import csv

from jaisalab.utils import get_time_stamp_as_string

class Plotter():
    def __init__(self, fdir, data_dir=None, dtype='np', savefig=True, **kwargs):
        if data_dir is None:
            data_dir = 'data/local/experiment'

        self.data_dir = data_dir
        self.fdir = fdir
        self.savefig = savefig
        self.fig_kwargs = kwargs

        if dtype == 'np':
            self.data = self.collect_data_as_np()
        elif dtype == 'pd':
            self.data = self.collect_data_as_pd()
        
        self.savedir = 'plots'

    def _collect_data(self):
        file = open(self.data_dir + f'/{self.fdir}/progress.csv')
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
        if self.savefig:
            try: 
                plt.savefig(f'plots/{self.fdir}_returns')
            except FileNotFoundError:
                os.mkdir('plots/')
                plt.savefig(f'plots/{self.fdir}_returns')

    def plot_entropy(self):
        pass
    def plot_constraints(self):
        pass
    