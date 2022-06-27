from jaisalab.experiments.trpo_backlog import trpo_inv_mng_backlog
from jaisalab.experiments.cpo_backlog import cpo_inv_mng_backlog
from jaisalab.metrics import Plotter

def test_trpo_backlog():
    trpo_inv_mng_backlog(seed=1)

def test_cpo_backlog():
    cpo_inv_mng_backlog(seed=1)

def plot_experiment():
    plotter = Plotter()
    plotter.plot_returns()
    plotter.plot_constraint_vals()

if __name__ == '__main__':
    #test_trpo_backlog()
    #test_cpo_backlog()
    plot_experiment()