"""
CPO on Inventory Management Backlog Environment. 
Author: Jaime Sabal
"""
import os 

#misc
import torch
from dowel import logger, StdOutput

#jaisalab
from jaisalab.utils.env import env_setup
from jaisalab.envs.inventory_management import InvManagementBacklogEnv
from jaisalab.algos import CPO
from jaisalab.safety_constraints import InventoryConstraints

#garage
import garage
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage import Trainer, wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, WorkerFactory
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction


@wrap_experiment
def cpo_inv_mng_backlog(ctxt=None, seed=1):
    """Train TRPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    # set up the dowel logger
    log_dir = os.path.join(os.getcwd(), 'data')
    ctxt = garage.experiment.SnapshotConfig(snapshot_dir=log_dir,
                                            snapshot_mode='last',
                                            snapshot_gap=1)

    # log to stdout
    logger.add_output(StdOutput())

    #set seed and define environment
    set_seed(seed)
    env = InvManagementBacklogEnv()
    env = env_setup(env) #set up environment

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(64, 64),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    #need to specify a worker factory to create sampler
    worker_factory = WorkerFactory(max_episode_length=env.max_episode_length)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           worker_factory=worker_factory)

    safety_constraint = InventoryConstraints()

    algo = CPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                safety_constraint=safety_constraint,
                sampler=sampler,
                discount=0.99,
                center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=500, batch_size=1024)