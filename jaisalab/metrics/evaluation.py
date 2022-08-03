#misc
import torch 

#garage
from garage.experiment import Snapshotter

#jaisalab

class Evaluator(object):
    """Class used to evaluate trained RL policies. Makes 
    use of garage's Snapshotter instance to extract the logged
    data by calling cloudpickle behind the scenes."""

    def __init__(self) -> None:
        pass

    def evaluate(self, snapshot_dir, n_epochs):
        pass

    def num_constraint_violations(self):
        pass