from jaisalab.experiments.backlog import trpo_backlog, cpo_backlog
from jaisalab.metrics import Plotter

def test_trpo_backlog():
    trpo_backlog(seed=1)

def test_cpo_backlog():
    cpo_backlog(seed=1)

def plot_experiment():
    fdir = ['cpo_backlog', 'trpo_backlog']
    plotter = Plotter(fdir=fdir)
    plotter.plot_returns()
    plotter.plot_constraint_vals()
    plotter.plot_costs()

if __name__ == '__main__':
    #test_trpo_backlog()
    #test_cpo_backlog()
    plot_experiment()