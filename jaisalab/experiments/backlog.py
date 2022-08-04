"""
Solving the Inventory Management Backlog Environment with a variety of 
Constrained RL algorithms.
Author: Jaime Sabal
"""
import os 
import numpy as np

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
from jaisalab.value_functions import GaussianValueFunction, QRValueFunction
from jaisalab.policies import SemiImplicitPolicy, GaussianPolicy

#garage
import garage
from garage import Trainer, wrap_experiment
from garage.experiment.deterministic import set_seed

@wrap_experiment
def cpo_backlog(ctxt=None, seed=1, n_epochs=800):
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

    safety_constraint = SoftInventoryConstraint(baseline=safety_baseline)

    sampler = SamplerSafe(agents=policy,
                          envs=env, 
                          max_episode_length=env.spec.max_episode_length, 
                          worker_args={'safety_constraint': safety_constraint})

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
def trpo_backlog(ctxt=None, seed=1, n_epochs=800):
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
def saute_trpo_backlog(ctxt=None, seed=1, n_epochs=800):
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


@wrap_experiment
def dcpo_backlog(ctxt=None, seed=1, n_epochs=800):
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

    value_function = QRValueFunction(env_spec=env.spec,
                                        N=102,
                                        hidden_sizes=(64, 64),
                                        hidden_nonlinearity=torch.tanh,
                                        output_nonlinearity=None)

    safety_baseline = QRValueFunction(env_spec=env.spec,
                                         N=102, 
                                         hidden_sizes=(64, 64),                                        
                                         hidden_nonlinearity=torch.tanh,
                                         output_nonlinearity=None)

    safety_constraint = SoftInventoryConstraint(baseline=safety_baseline)

    sampler = SamplerSafe(agents=policy,
                          envs=env, 
                          max_episode_length=env.spec.max_episode_length, 
                          worker_args={'safety_constraint': safety_constraint})

    algo = CPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               safety_constraint=safety_constraint,
               sampler=sampler,
               discount=0.99,
               center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=n_epochs, batch_size=1024)
