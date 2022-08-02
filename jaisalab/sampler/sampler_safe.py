"""Sampler that runs workers in the main process."""
#misc
import psutil
import torch

#garage
from garage.experiment.deterministic import get_seed
from garage.sampler import LocalSampler
from garage.sampler.worker_factory import WorkerFactory

#jaisalab
from jaisalab.sampler.safe_worker import SafeWorker
from jaisalab.safety_constraints import SoftInventoryConstraint, BaseConstraint
from jaisalab import SafeEpisodeBatch
from jaisalab.value_functions import GaussianValueFunction


class SamplerSafe(LocalSampler):
    """Sampler that runs workers in the main process.

    Inherits from LocalSampler, which runs everything in the same 
    process and thread as where it was called from, and provides further 
    functionalities for algorithms that incorporate safety.

    The sampler need to be created either from a worker factory or from
    parameters which can construct a worker factory. See the __init__ method
    of WorkerFactory for the detail of these parameters.

    Args:
        agents (Policy or List[Policy]): Agent(s) to use to sample episodes.
            If a list is passed in, it must have length exactly
            `worker_factory.n_workers`, and will be spread across the
            workers.
        envs (Environment or List[Environment]): Environment from which
            episodes are sampled. If a list is passed in, it must have length
            exactly `worker_factory.n_workers`, and will be spread across the
            workers.
        worker_factory (WorkerFactory): Pickleable factory for creating
            workers. Should be transmitted to other processes / nodes where
            work needs to be done, then workers should be constructed
            there. Either this param or params after this are required to
            construct a sampler.
        max_episode_length(int): Params used to construct a worker factory.
            The maximum length episodes which will be sampled.
        is_tf_worker (bool): Whether it is workers for TFTrainer.
        seed(int): The seed to use to initialize random number generators.
        n_workers(int): The number of workers to use.
        worker_class(type): Class of the workers. Instances should implement
            the Worker interface.
        worker_args (dict or None): Additional arguments that should be passed
            to the worker.

    """
    def __init__(self,
                 agents, 
                 envs,        
                 *,  # After this require passing by keyword.
                 max_episode_length=None,
                 is_tf_worker=False,
                 seed=get_seed(),
                 n_workers=psutil.cpu_count(logical=False),
                 worker_class=SafeWorker,
                 worker_args={'safety_constraint':None}):
        
        #impose a safety constraint to the environment
        if worker_args['safety_constraint'] is None:
            if hasattr(envs, 'supply_capacity'):
                safety_baseline = GaussianValueFunction(env_spec=envs.spec,
                                        hidden_sizes=(64, 64),
                                        hidden_nonlinearity=torch.tanh,
                                        output_nonlinearity=None)
                worker_args['safety_constraint'] = SoftInventoryConstraint(baseline=safety_baseline)
            else: 
                raise AttributeError("Please specify a safety constraint for the environment")
                
        if not isinstance(worker_args['safety_constraint'], BaseConstraint):
            raise TypeError("The safety constraint must be an instance of the BaseConstraint class.")
        if max_episode_length is None: 
            worker_factory = WorkerFactory(max_episode_length=envs.spec.max_episode_length, 
                                           worker_class=worker_class, worker_args=worker_args)
        else: 
            worker_factory=None
        super().__init__(agents, 
                         envs, 
                         worker_factory=worker_factory,
                         max_episode_length=max_episode_length, 
                         is_tf_worker=is_tf_worker, 
                         seed=seed,
                         n_workers=n_workers,
                         worker_class=worker_class,
                         worker_args=worker_args)
        
    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Collect at least a given number transitions (timesteps).

        Args:
            itr(int): The current iteration number. Using this argument is
                deprecated.
            num_samples (int): Minimum number of transitions / timesteps to
                sample.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            EpisodeBatch: The batch of collected episodes.

        """
        self._update_workers(agent_update, env_update)
        batches = []
        completed_samples = 0
        while True:
            for worker in self._workers:
                batch = worker.rollout()
                completed_samples += len(batch.actions)
                batches.append(batch)
                if completed_samples >= num_samples:
                    samples = SafeEpisodeBatch.concatenate(*batches)
                    self.total_env_steps += sum(samples.lengths)
                    return samples

    
