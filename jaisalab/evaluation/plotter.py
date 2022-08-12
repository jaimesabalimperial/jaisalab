#misc
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as ani

#jaisalab
from jaisalab.evaluation.plotter_base import BasePlotter
from jaisalab.evaluation.evaluator import SeedEvaluator

class RLPlotter(BasePlotter):
    """Plotting functionalities for evaluation of constrained RL 
    algorithms implemented through the jaisalab and garage frameworks.
    
    Args: 
        plot_latest (bool): Wether to plot the latest results found in the default
          data directory --> 'data/local/experiment'. If the default directory is specified 
          by the user, this must also be specified in the data_dir argument. 
        
        fdir (str, list): Name of the experiment results directory within the data directory.
          If a list or tuple, all the experiments specified will be plotted. If None, plot_latest
          should be True. *NOTE*: It is assumed that fdirs (i.e. experiment functions) are named 
          in the format '{algorithm}_{environment}'. 

        data_dir (str): Data directory. Default = 'data/local/experiment'.

        dtype (str): Datatype of parsed results. Options=('np'), Default='np'.

        savefig (bool): Wether to save plotted figures (By default these are saved to 'plots' 
          directory within the current working directory). 
    """
    def __init__(self, get_latest=True, fdir=None, data_dir='data/local/experiment', 
                 dtype='np', savefig=True, use_legend=True, **kwargs):
        super().__init__(get_latest=get_latest, fdir=fdir, data_dir=data_dir, 
                         dtype=dtype, savefig=savefig, use_legend=use_legend, **kwargs)

    def plot_returns(self, title=None, use_legend=True, return_lim=None, **kwargs):
        """Plot progression of returns throughout epochs."""
        plot_kwargs = kwargs if kwargs is not None else {}
        y_column = 'Evaluation/AverageReturn'
        ylabel = 'Average Return'
    
        std_column = 'Evaluation/StdReturn'
        x_array, y_array, std_array = self.get_data_arrays(y_column, std_column)

        if return_lim is not None: 
            plot_kwargs['hline_label'] = 'max return'

        self.plot(x_array, y_array, ylabel, std=std_array, 
                  title=title, use_legend=use_legend, 
                  hline=return_lim, **plot_kwargs)
        self.savefig(flag=1)

    def plot_costs(self, title=None, use_legend=True, cost_lim=None, **kwargs):
        """Plot progression of costs throughout epochs."""
        plot_kwargs = kwargs if kwargs is not None else {}
        y_column = 'Evaluation/AverageSafetyReturn'
        std_column = 'Evaluation/StdSafetyReturn'
        ylabel = 'Average Costs'

        x_array, y_array, std_array = self.get_data_arrays(y_column, std_column)
        
        if cost_lim is not None: 
            plot_kwargs['hline_label'] = 'max cost'

        self.plot(x_array, y_array, ylabel, std=std_array, 
                  title=title, use_legend=use_legend, 
                  hline=cost_lim, **plot_kwargs)
        self.savefig(flag=6)

    def plot_constraint_vals(self, title=None, use_legend=True, 
                             cost_lim=None, **kwargs):
        """Plot progression of constraint values (i.e. average discounted 
        costs) throughout learning."""
        plot_kwargs = kwargs if kwargs is not None else {}
        y_column = 'Evaluation/AverageDiscountedSafetyReturn'
        ylabel = 'Constraint Value'

        if cost_lim is not None: 
            plot_kwargs['hline_label'] = 'max cost'

        if self.std_data is not None: 
            x_array, y_array, std_array = self.get_data_arrays(y_column)
            self.plot(x_array, y_array, ylabel, std=std_array, 
                      title=title, use_legend=use_legend, **plot_kwargs)
        else: 
            x_array, y_array = self.get_data_arrays(y_column)
            self.plot(x_array, y_array, ylabel, 
                      title=title, use_legend=use_legend, **plot_kwargs)

        self.savefig(flag=3)
    
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
        mean_returns_array, std_returns_array = self.get_distribution_data()
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
        self.savefig(flag=7, animator=animator)

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
        self.savefig(flag=8)

    def plot_quantiles_progression(self, Vmin=-800, Vmax=800, interval=20, **kwargs):
        """Plot progression of quantile value distribution throughout learning."""
        quantile_probs, quantile_vals = self.get_quantiles_data(Vmin, Vmax)

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
        self.savefig(flag=9, animator=animator)
    
    def plot_evaluation(self, seed_dir):
        seed_evaluator = SeedEvaluator(seed_dir)
        

    
    