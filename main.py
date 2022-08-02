from torch import seed
from jaisalab.experiments.backlog import (trpo_backlog, cpo_backlog, 
                                          saute_trpo_backlog, dcpo_backlog)
from jaisalab.metrics import Plotter

def test_trpo_backlog():
    trpo_backlog(seed=1)

def test_cpo_backlog():
    cpo_backlog(seed=1)

def test_saute_trpo_backlog():
    saute_trpo_backlog(seed=1)

def test_dcpo_backlog():
    dcpo_backlog(seed=1)


def plot_experiment():
    fdir = ['dcpo_backlog_2']#'trpo_backlog_70']#, 'cpo_backlog_2', 'saute_trpo_backlog_19', 'dcpo_backlog_1']
    plotter = Plotter(fdir=fdir)
    #plotter.plot_returns()
    #plotter.plot_constraint_vals()
    #plotter.plot_costs()
    #plotter.plot_gaussian_progression(num_points=400)
    #plotter.plot_final_distribution()
    plotter.plot_quantiles_progression()

if __name__ == '__main__':
    #test_trpo_backlog()
    #test_cpo_backlog()
    #test_saute_trpo_backlog()
    #test_dcpo_backlog()
    plot_experiment()