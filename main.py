from torch import seed
from jaisalab.experiments.backlog import (trpo_backlog, cpo_backlog, 
                                          saute_trpo_backlog, dcpo_backlog)
from jaisalab.evaluation import RLPlotter

def test_trpo_backlog(seed):
    trpo_backlog(seed=seed)

def test_cpo_backlog(seed):
    cpo_backlog(seed=seed)

def test_saute_trpo_backlog(seed):
    saute_trpo_backlog(seed=seed)

def test_dcpo_backlog(seed):
    dcpo_backlog(seed=seed)


def plot_experiment():
    #fdir = ['trpo_backlog_1', 'cpo_backlog_1', 'saute_trpo_backlog_1', 'dcpo_backlog_1']
    plotter = RLPlotter(fdir='dcpo_backlog_1')
    #plotter.plot_returns()
    #plotter.plot_constraint_vals()
    #plotter.plot_costs()
    #plotter.plot_gaussian_progression(num_points=400)
    #plotter.plot_final_distribution()
    plotter.plot_quantiles_progression(interval=10)

if __name__ == '__main__':
    seed_val = 3
    #test_trpo_backlog(seed=seed_val)
    test_cpo_backlog(seed=seed_val)
    #test_saute_trpo_backlog(seed=seed_val)
    #test_dcpo_backlog(seed=seed_val)
    #plot_experiment()