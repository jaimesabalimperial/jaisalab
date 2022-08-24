#misc
import torch
from dowel import logger, StdOutput

#jaisalab
from jaisalab.utils.env import env_setup
from jaisalab.envs.inventory_management import InvManagementBacklogEnv, SauteInvManagementBacklogEnv
from jaisalab.algos.cpo import CPO
from jaisalab.algos.trpo_safe import SafetyTRPO
from jaisalab.algos.dcpo import DCPO
from jaisalab.safety_constraints import SoftInventoryConstraint
from jaisalab.sampler.sampler_safe import SamplerSafe
from jaisalab.value_functions import GaussianValueFunction, QRValueFunction
from jaisalab.policies import GaussianPolicy
from jaisalab.evaluation import RLPlotter

#garage
from garage import Trainer, wrap_experiment
from garage.experiment.deterministic import set_seed

seeds_list = [1]

for SEED in seeds_list:
    @wrap_experiment(log_dir=f'data{SEED}/cpo_backlog')
    def cpo_backlog(ctxt, seed=1, n_epochs=800):
        """Train CPO with InvManagementBacklogEnv environment.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
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

    @wrap_experiment(log_dir=f'data{SEED}/trpo_backlog')
    def trpo_backlog(ctxt, seed=1, n_epochs=800):
        """Train TRPO with IMP environment.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
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

    @wrap_experiment(log_dir=f'data{SEED}/saute_trpo_backlog')
    def saute_trpo_backlog(ctxt, seed=1, n_epochs=800):
        """Train TRPO with InvertedDoublePendulum-v2 environment.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
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


    @wrap_experiment(log_dir=f'data{SEED}/dcpo_eps=0.15_backlog')
    def dcpo_beta_backlog(ctxt, seed=1, n_epochs=800):
        """Train CPO with InvManagementBacklogEnv environment.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
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
                                        Vmin=-800, 
                                        Vmax=800.,
                                        N=102,
                                        hidden_sizes=(64, 64),
                                        hidden_nonlinearity=torch.tanh,
                                        output_nonlinearity=None)

        safety_baseline = QRValueFunction(env_spec=env.spec,
                                        Vmin=0, 
                                        Vmax=60.,
                                        N=102, 
                                        hidden_sizes=(64, 64),                                        
                                        hidden_nonlinearity=torch.tanh,
                                        output_nonlinearity=None)

        safety_constraint = SoftInventoryConstraint(baseline=safety_baseline)

        sampler = SamplerSafe(agents=policy,
                            envs=env, 
                            max_episode_length=env.spec.max_episode_length, 
                            worker_args={'safety_constraint': safety_constraint})

        algo = DCPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                safety_constraint=safety_constraint,
                sampler=sampler,
                discount=0.99,
                center_adv=False, 
                safety_margin=0.15, 
                beta=1., 
                dist_penalty=True) 

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=1024)

    @wrap_experiment(log_dir=f'data{SEED}/dcpo_ablation_backlog')
    def dcpo_ablation_backlog(ctxt, seed=1, n_epochs=800):
        """Train CPO with InvManagementBacklogEnv environment.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
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
                                        Vmin=-800,
                                        Vmax=800,
                                        N=102,
                                        hidden_sizes=(64, 64),
                                        hidden_nonlinearity=torch.tanh,
                                        output_nonlinearity=None)

        safety_baseline = QRValueFunction(env_spec=env.spec,
                                        Vmin=0, 
                                        Vmax=60.,
                                        N=102, 
                                        hidden_sizes=(64, 64),                                        
                                        hidden_nonlinearity=torch.tanh,
                                        output_nonlinearity=None)

        safety_constraint = SoftInventoryConstraint(baseline=safety_baseline)

        sampler = SamplerSafe(agents=policy,
                            envs=env, 
                            max_episode_length=env.spec.max_episode_length, 
                            worker_args={'safety_constraint': safety_constraint})

        algo = DCPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                safety_constraint=safety_constraint,
                sampler=sampler,
                discount=0.99,
                center_adv=False, 
                dist_penalty=False) #running ablation

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=1024)

    def train_seed():
        #trpo_backlog(seed=SEED)
        #cpo_backlog(seed=SEED)
        #saute_trpo_backlog(seed=SEED)
        dcpo_beta_backlog(seed=SEED)

    if __name__ == '__main__':
        train_seed()
        #trpo_backlog(seed=SEED)
        #cpo_backlog(seed=SEED)
        #saute_trpo_backlog(seed=SEED)
        #dcpo_backlog(seed=SEED)
        #eps_labels = [r'$\epsilon = 0.0', r'$\epsilon = 0.5$', r'$\beta = 0.97$']
        #plotter = RLPlotter(fdir=['dcpo_eps=0.0_backlog', 'dcpo_eps=0.5_backlog', 'dcpo_eps=0.97_backlog'], data_dir='data1')
        #final_returns = plotter.plot_returns(figsize=(10,6), fontsize=12, custom_labels=eps_labels)
        #final_costs = plotter.plot_costs(cost_lim=15, use_legend=False, figsize=(10,6), fontsize=12)