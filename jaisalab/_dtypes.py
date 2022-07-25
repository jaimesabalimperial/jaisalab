from garage import EpisodeBatch
import numpy as np
from garage.np import (concat_tensor_dict_list, pad_batch_array,
                       slice_nested_dict, stack_tensor_dict_list)

class SafeEpisodeBatch(EpisodeBatch):
    def __init__(self, env_spec, episode_infos, observations, last_observations,
                 actions, rewards, safety_rewards, env_infos, agent_infos, step_types, lengths):

        object.__setattr__(self, 'safety_rewards', safety_rewards) #needed to add safety_rewards 

        super().__init__(env_spec, episode_infos, observations, last_observations, 
                         actions, rewards, env_infos, agent_infos, step_types, lengths)
    
    @property
    def padded_safety_rewards(self):
        """Padded rewards.

        Returns:
            np.ndarray: Padded rewards with shape of
                :math:`(N, max_episode_length)`.

        """
        return pad_batch_array(self.safety_rewards, self.lengths,
                               self.env_spec.max_episode_length)
  
    
    def split(self):
        """Split an EpisodeBatch into a list of EpisodeBatches.

        The opposite of concatenate.

        Returns:
            list[EpisodeBatch]: A list of EpisodeBatches, with one
                episode per batch.

        """
        episodes = []
        for i, (start, stop) in enumerate(self._episode_ranges()):
            eps = SafeEpisodeBatch(
                    env_spec=self.env_spec,
                    episode_infos=slice_nested_dict(self.episode_infos_by_episode,
                                                    i, i + 1),
                    observations=self.observations[start:stop],
                    last_observations=np.asarray([self.last_observations[i]]),
                    actions=self.actions[start:stop],
                    rewards=self.rewards[start:stop],
                    safety_rewards=self.safety_rewards[start:stop],
                    env_infos=slice_nested_dict(self.env_infos, start, stop),
                    agent_infos=slice_nested_dict(self.agent_infos, start, stop),
                    step_types=self.step_types[start:stop],
                    lengths=np.asarray([self.lengths[i]]))
            episodes.append(eps)

        return episodes
    
    @classmethod
    def concatenate(cls, *batches):
        """Create a EpisodeBatch by concatenating EpisodeBatches.

        Args:
            batches (list[EpisodeBatch]): Batches to concatenate.

        Returns:
            EpisodeBatch: The concatenation of the batches.

        """
        if __debug__:
            for b in batches:
                assert (set(b.env_infos.keys()) == set(
                    batches[0].env_infos.keys()))
                assert (set(b.agent_infos.keys()) == set(
                    batches[0].agent_infos.keys()))
        env_infos = {
            k: np.concatenate([b.env_infos[k] for b in batches])
            for k in batches[0].env_infos.keys()
        }
        agent_infos = {
            k: np.concatenate([b.agent_infos[k] for b in batches])
            for k in batches[0].agent_infos.keys()
        }
        episode_infos = {
            k: np.concatenate([b.episode_infos_by_episode[k] for b in batches])
            for k in batches[0].episode_infos_by_episode.keys()
        }
        return cls(
            episode_infos=episode_infos,
            env_spec=batches[0].env_spec,
            observations=np.concatenate(
                [batch.observations for batch in batches]),
            last_observations=np.concatenate(
                [batch.last_observations for batch in batches]),
            actions=np.concatenate([batch.actions for batch in batches]),
            rewards=np.concatenate([batch.rewards for batch in batches]),
            safety_rewards=np.concatenate([batch.safety_rewards for batch in batches]), #added this line to garage EpisodeBatch
            env_infos=env_infos,
            agent_infos=agent_infos,
            step_types=np.concatenate([batch.step_types for batch in batches]),
            lengths=np.concatenate([batch.lengths for batch in batches]))