"""TRPO on Inventory Management Environment
Author - Jaime Sabal"""
import os 

#misc
import torch

#jaisalab
from jaisalab.utils._env import env_setup
from jaisalab.envs.inventory_management import InvManagementBacklogEnv

#garage
import garage
from garage.tf.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage import Trainer
from dowel import logger, StdOutput
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, WorkerFactory
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction


@wrap_experiment
def trpo_inv_mng_backlog(ctxt=None, seed=1):
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
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    #need to specify a worker factory to create sampler
    worker_factory = WorkerFactory(max_episode_length=env.max_episode_length)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           worker_factory=worker_factory)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                sampler=sampler,
                discount=0.99,
                center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100, batch_size=1024)
