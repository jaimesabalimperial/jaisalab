import pytest

from jaisalab.evaluation.plotter import RLPlotter

single_data_dir = 'test_data/data1/'
multiple_data_dirs = ['test_data/data1', 'test_data/data2/', 'test_data/data3/']

single_fdir = 'dcpo_backlog'
multiple_fdirs = ['trpo_backlog', 'cpo_backlog', 
                  'saute_trpo_backlog', 'dcpo_backlog']

def test_plotter_init():
    #single data_dir with single fdir
    plotter = RLPlotter(fdir=single_fdir, data_dir=single_data_dir, savefig=False)

    #single data_dir with multiple fdirs
    plotter = RLPlotter(fdir=multiple_fdirs, data_dir=single_data_dir, savefig=False)

    #multiple data_dir with single fdir
    plotter = RLPlotter(fdir=single_fdir, data_dir=multiple_data_dirs, savefig=False)

    #multiple data_dir with multiple fdirs
    plotter = RLPlotter(fdir=multiple_fdirs, data_dir=multiple_data_dirs, savefig=False)

def test_plots():
    #single data_dir with single fdir
    plotter = RLPlotter(fdir=single_fdir, data_dir=single_data_dir, savefig=False)
    plotter.plot_returns()
    plotter.plot_costs()
    plotter.plot_quantiles_progression()

    #single data_dir with multiple fdirs
    plotter = RLPlotter(fdir=multiple_fdirs, data_dir=single_data_dir, savefig=False)
    plotter.plot_returns()
    plotter.plot_costs()

    #multiple data_dir with single fdir
    plotter = RLPlotter(fdir=single_fdir, data_dir=multiple_data_dirs, savefig=False)
    plotter.plot_returns()
    plotter.plot_costs()

    #multiple data_dir with multiple fdirs
    plotter = RLPlotter(fdir=multiple_fdirs, data_dir=multiple_data_dirs, savefig=False)
    plotter.plot_returns()
    plotter.plot_costs()

if __name__ == '__main__':
    test_plotter_init()
    test_plots()