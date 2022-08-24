#misc
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as ani
import seaborn as sns
import inspect

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
        y_column = 'Evaluation/AverageReturn'
        ylabel = 'Average Return'
    
        std_column = 'Evaluation/StdReturn'
        x_array, y_array, std_array = self.get_data_arrays(y_column, std_column)

        if return_lim is not None: 
            kwargs['hline_label'] = 'max return'

        self.plot(x_array, y_array, ylabel, std=std_array, 
                  title=title, use_legend=use_legend, 
                  hline=return_lim, **kwargs)
        self.savefig(flag=1)

    def plot_costs(self, title=None, use_legend=True, cost_lim=None, **kwargs):
        """Plot progression of costs throughout epochs."""
        y_column = 'Evaluation/AverageSafetyReturn'
        std_column = 'Evaluation/StdSafetyReturn'
        ylabel = 'Average Costs'

        x_array, y_array, std_array = self.get_data_arrays(y_column, std_column)
        
        if cost_lim is not None: 
            kwargs['hline_label'] = 'max cost'

        self.plot(x_array, y_array, ylabel, std=std_array, 
                  title=title, use_legend=use_legend, 
                  hline=cost_lim, **kwargs)
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
            self.plot(x_array, y_array, ylabel, title=title, 
                      use_legend=use_legend, **plot_kwargs)

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
    
    def plot_quantiles_progression(self, Vmin=-800, Vmax=800, interval=20, metric='task', 
                                   training_steps = None, **kwargs):
        """Plot progression of quantile value distribution throughout learning."""
        assert metric in ['task', 'safety'], "Metric to be plotted must either be 'task' or 'safety'."

        if 'fontsize' in kwargs.keys():
            plt.rcParams.update({'font.size': kwargs['fontsize']})

        #retrieve quantiles data    
        quantile_probs, quantile_vals = self.get_quantiles_data(Vmin, Vmax, metric)

        if metric == 'task':
            xlabel = 'Value'
            color = 'royalblue'
            edge_color = 'navy'
        else: 
            xlabel = 'Constraint Value'
            color = 'red'
            edge_color = 'maroon'
        
        width = (Vmax - Vmin) / 100 #bar width

        fig_kwargs = {k:v for k,v in kwargs.items() if k in inspect.getargspec(plt.figure)[0]}
        fig = plt.figure(**fig_kwargs)

        if training_steps is not None:
            for step in training_steps:
                fig = plt.figure(**fig_kwargs)
                plt.ylim(0,1)
                p = plt.bar(quantile_vals, quantile_probs[step], 
                            color=color, edgecolor=edge_color, width=width)
                plt.title(f'epoch: {step+1}')
                plt.xlabel(xlabel)
                plt.ylabel('Probability')
                plt.savefig(f'plots/{metric}_epoch_{step+1}.png')
        else:
            fig = plt.figure(**fig_kwargs)
            def dist_progress(i):
                fig.clear()
                plt.ylim(0,1)
                p = plt.bar(quantile_vals, quantile_probs[i*interval], 
                            color=color, edgecolor=edge_color, width=width)
                plt.title(f'epoch: {i*interval}')
                plt.xlabel(xlabel)
                plt.ylabel('Probability')

            time_interval = 5000 / len(quantile_probs)
            save_count = len(quantile_probs) // interval
            animator = ani.FuncAnimation(fig, dist_progress, interval=time_interval, save_count=save_count)
            self.savefig(flag=9, animator=animator)
    
    def plot_evaluation(self, seed_dir, experiments, labels, **kwargs):
        """Plot the evaluation results in terms of the average normalised 
        returns and costs throughout the test epochs. The normalisation of 
        the costs is done in terms of the maximum allowed cost defined in the 
        safety constraint class inputted into the algorithm. The normalisation 
        of the returns is done in terms of the average cost of DCPO, since we 
        want to compare it against the other algorithms.
        
        Args: 
            seed_dir (str): Directory containing seed folders.
            
            experiments (dict): Dictionary containing keys as plot labels 
                and values as the experiment folders we want to plot.
        """
        if 'fontsize' in kwargs.keys():
            plt.rcParams.update({'font.size': kwargs['fontsize']})

        conv_dict = {exp:label for exp,label in zip(experiments, labels)}
        seed_evaluator = SeedEvaluator(seed_dir, override=False)
        data = seed_evaluator.get_evaluation()

        exps_to_remove = np.unique([exp for exp in data.experiment if exp not in experiments])
        plot_data = data.copy()

        for exp in exps_to_remove:
            plot_data = plot_data.drop(data[(data['experiment'] == exp)].index)

        plot_labels = [conv_dict[exp] for exp in plot_data.experiment]
        plot_data.experiment = plot_labels

        fig, ax = plt.subplots(figsize=(10,7))
        g = sns.boxplot(x="experiment", y="metric", hue="tag",
                        data=plot_data, palette="deep")
        g.axhline(1.0, color='black', linestyle='dashed')
        ax.set_ylim(bottom=0.0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:])
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        self.savefig(flag=10)
    