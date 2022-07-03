from collections import defaultdict

import numpy as np

from garage import EpisodeBatch, StepType
from garage.sampler import DefaultWorker

from jaisalab import SafeEpisodeBatch

class SafeWorker(DefaultWorker):
    """Worker for environments that incorporate safety."""
    def __init__(self, *, safety_constraint, seed, max_episode_length, worker_number):
        self.safety_constraint = safety_constraint
        super().__init__(seed=seed, 
                         max_episode_length=max_episode_length, 
                         worker_number=worker_number)
    
    def collect_episode(self):
        """Collect the current episode, clearing the internal buffer.

        Returns:
            EpisodeBatch: A batch of the episodes completed since the last call
                to collect_episode().

        """
        observations = self._observations
        self._observations = []
        last_observations = self._last_observations
        self._last_observations = []

        actions = []
        rewards = []
        env_infos = defaultdict(list)
        step_types = []

        for es in self._env_steps:
            actions.append(es.action)
            rewards.append(es.reward)
            step_types.append(es.step_type)
            for k, v in es.env_info.items():
                env_infos[k].append(v)
        self._env_steps = []

        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)

        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)

        episode_infos = self._episode_infos
        self._episode_infos = defaultdict(list)
        for k, v in episode_infos.items():
            episode_infos[k] = np.asarray(v)

        lengths = self._lengths
        self._lengths = []
        
        path = EpisodeBatch(env_spec=self.env.spec,
                            episode_infos=episode_infos,
                            observations=np.asarray(observations),
                            last_observations=np.asarray(last_observations),
                            actions=np.asarray(actions),
                            rewards=np.asarray(rewards),
                            step_types=np.asarray(step_types, dtype=StepType),
                            env_infos=dict(env_infos),
                            agent_infos=dict(agent_infos),
                            lengths=np.asarray(lengths, dtype='i'))
                            
        #consider safety constraint
        safety_rewards = self.safety_constraint.evaluate(path)

        return SafeEpisodeBatch(env_spec=self.env.spec,
                            episode_infos=episode_infos,
                            observations=np.asarray(observations),
                            last_observations=np.asarray(last_observations),
                            actions=np.asarray(actions),
                            rewards=np.asarray(rewards),
                            safety_rewards=np.asarray(safety_rewards),
                            step_types=np.asarray(step_types, dtype=StepType),
                            env_infos=dict(env_infos),
                            agent_infos=dict(agent_infos),
                            lengths=np.asarray(lengths, dtype='i'))
