import pytest

#misc
from jaisalab.experiments import (trpo_backlog, cpo_backlog, 
                                  saute_trpo_backlog, dcpo_backlog)

def test_cpo_backlog():
    cpo_backlog(seed=1, n_epochs=2)

def test_trpo_backlog():
    trpo_backlog(seed=1, n_epochs=2)

def test_saute_trpo_backlog():
    saute_trpo_backlog(seed=1, n_epochs=2)

def test_dcpo_backlog():
    dcpo_backlog(seed=1, n_epochs=2)

if __name__ == '__main__':
    test_trpo_backlog()
    test_cpo_backlog()
    test_saute_trpo_backlog()
    test_dcpo_backlog()

