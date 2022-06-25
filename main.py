from jaisalab.experiments.trpo_backlog import trpo_inv_mng_backlog
from jaisalab.experiments.cpo_backlog import cpo_inv_mng_backlog
from jaisalab.metrics import Plotter

def test_trpo_backlog():
    trpo_inv_mng_backlog(seed=1)

def test_cpo_backlog():
    cpo_inv_mng_backlog(seed=1)

def plot_experiment():
    fdir = 'cpo_inv_mng_backlog_34'
    plotter = Plotter(fdir = fdir)
    plotter.plot_returns()
    
if __name__ == '__main__':
    #test_cpo_backlog()
    plot_experiment()