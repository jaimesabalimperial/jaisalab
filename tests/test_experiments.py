import pytest
from jaisalab.experiments import (trpo_backlog, cpo_backlog, 
                                  saute_trpo_backlog)

if __name__ == '__main__':
    trpo_backlog(seed=1)
    cpo_backlog(seed=1)
    saute_trpo_backlog(seed=1)

