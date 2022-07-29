import pytest

import os 

#misc
import torch
from dowel import logger, StdOutput

#jaisalab
from jaisalab.utils.env import env_setup
from jaisalab.envs.inventory_management import InvManagementBacklogEnv, SauteInvManagementBacklogEnv
from jaisalab.algos.cpo import CPO
from jaisalab.algos.trpo import SafetyTRPO
from jaisalab.safety_constraints import SoftInventoryConstraint
from jaisalab.sampler.sampler_safe import SamplerSafe
from jaisalab.value_functions import GaussianValueFunction
from jaisalab.policies import SemiImplicitPolicy, GaussianPolicy

#garage
import garage
from garage import Trainer, wrap_experiment
from garage.experiment.deterministic import set_seed

@wrap_experiment
def test_cpo_backlog(ctxt=None, seed=1, n_epochs=600):
    """Train CPO with InvManagementBacklogEnv environment.

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

    policy = GaussianPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianValueFunction(env_spec=env.spec,
                                              hidden_sizes=(64, 64),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    safety_baseline = GaussianValueFunction(env_spec=env.spec,
                                            hidden_sizes=(64, 64),                                        
                                            hidden_nonlinearity=torch.tanh,
                                            output_nonlinearity=None)

    sampler = SamplerSafe(agents=policy,
                          envs=env)

    safety_constraint = SoftInventoryConstraint(baseline=safety_baseline)

    algo = CPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               safety_constraint=safety_constraint,
               sampler=sampler,
               discount=0.99,
               center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=n_epochs, batch_size=1024)

@wrap_experiment
def test_trpo_backlog(ctxt=None, seed=1, n_epochs=600):
    """Train TRPO with IMP environment.

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

    policy = SemiImplicitPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               noise_dim=10, 
                               noise_num=5,
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianValueFunction(env_spec=env.spec,
                                              hidden_sizes=(64, 64),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    safety_baseline = GaussianValueFunction(env_spec=env.spec,
                                        hidden_sizes=(64, 64),
                                        hidden_nonlinearity=torch.tanh,
                                        output_nonlinearity=None)

    safety_constraint = SoftInventoryConstraint(baseline=safety_baseline)

    sampler = SamplerSafe(agents=policy,
                          envs=env, 
                          max_episode_length=env.spec.max_episode_length, 
                          worker_args={'safety_constraint': safety_constraint})

    algo = SafetyTRPO(env_spec=env.spec,
                      policy=policy,
                      value_function=value_function,
                      sampler=sampler,
                      safety_constraint=safety_constraint,
                      discount=0.99,
                      center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=n_epochs, batch_size=1024)

@wrap_experiment
def test_saute_trpo_backlog(ctxt=None, seed=1, n_epochs=600):
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
    env = SauteInvManagementBacklogEnv()
    env = env_setup(env) #set up environment

    trainer = Trainer(ctxt)

    policy = GaussianPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    safety_baseline = GaussianValueFunction(env_spec=env.spec,
                                        hidden_sizes=(64, 64),
                                        hidden_nonlinearity=torch.tanh,
                                        output_nonlinearity=None)

    safety_constraint = SoftInventoryConstraint(baseline=safety_baseline)

    sampler = SamplerSafe(agents=policy,
                          envs=env, 
                          max_episode_length=env.spec.max_episode_length, 
                          worker_args={'safety_constraint': safety_constraint})

    algo = SafetyTRPO(env_spec=env.spec,
                      policy=policy,
                      value_function=value_function,
                      sampler=sampler,
                      safety_constraint=safety_constraint,
                      discount=0.99,
                      center_adv=False, 
                      is_saute=True)

    trainer.setup(algo, env)
    trainer.train(n_epochs=n_epochs, batch_size=1024)

if __name__ == '__main__':
    test_trpo_backlog(seed=1, n_epochs=5)
    test_cpo_backlog(seed=1, n_epochs=5)
    test_saute_trpo_backlog(seed=1, n_epochs=5)

