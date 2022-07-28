from torch import seed
from jaisalab.experiments.backlog import (trpo_backlog, cpo_backlog, 
                                          saute_trpo_backlog, iqn_trpo)
from jaisalab.metrics import Plotter

def test_trpo_backlog():
    trpo_backlog(seed=1)

def test_cpo_backlog():
    cpo_backlog(seed=1)

def test_saute_trpo_backlog():
    saute_trpo_backlog(seed=1)

def test_iqn_trpo():
    iqn_trpo(seed=1)

def plot_experiment():
    #fdir = ['trpo_backlog_1', 'cpo_backlog_2', 'saute_trpo_backlog_19']
    plotter = Plotter(fdir=None)
    plotter.plot_returns()
    #plotter.plot_constraint_vals()
    plotter.plot_costs()
    #plotter.plot_distribution_progression(num_points=400)
    plotter.plot_final_distribution()

if __name__ == '__main__':
    #test_trpo_backlog()
    test_cpo_backlog()
    #test_saute_trpo_backlog()
    #test_iqn_trpo()
    plot_experiment()