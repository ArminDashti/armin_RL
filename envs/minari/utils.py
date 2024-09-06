


def dataset_to_dict(dataset):
    episodes = []
    for episode_data in self.dataset.iterate_episodes():
        episode = []
        for step in range(episode_data.total_timesteps):
            sample = {}
            sample['observation'] = episode_data.observations[step, :]
            sample['action'] = episode_data.actions[step]
            sample['reward'] = episode_data.rewards[step]
            if (step == episode_data.total_timesteps-1):
                sample['next_observation'] = episode_data.observations[step]
            else:
                sample['next_observation'] = episode_data.observations[step+1]
                sample['termination'] = episode_data.terminations[step]
                sample['truncation'] = episode_data.truncations[step]
                sample['info'] = episode_data.infos['success'][step]
                episode.append(sample)
            episodes.append(episode)
                
    return episodes




    
    
def dataset_to_reply_memory(dataset):
    time_steps_num = self.time_steps_num
    episodes_num = self.episodes_num
    capacity = time_steps_num * episodes_num
    rm = reply_memory(capacity=capacity)
    
    dataset = self.raw_dataset()
    
    for episode in dataset:
        for step in episode:
            observations = step['observation']
            actions = step['action']
            rewards = step['reward']
            next_observations = step['next_observation']
            terminations = step['termination']
            truncations = step['truncation']
            infos = step['info']
            rm.push(observations=observations, actions=actions, rewards=rewards, truncations=truncations, next_observations=next_observations)
                
    return rm