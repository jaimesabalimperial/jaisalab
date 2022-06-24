from jaisalab.experiments.trpo_backlog import trpo_inv_mng_backlog
from jaisalab.experiments.cpo_backlog import cpo_inv_mng_backlog

def test_trpo_backlog():
    trpo_inv_mng_backlog(seed=1)

def test_cpo_backlog():
    cpo_inv_mng_backlog(seed=1)

if __name__ == '__main__':
    test_cpo_backlog()