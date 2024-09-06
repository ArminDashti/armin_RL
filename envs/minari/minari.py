import minari
import numpy as np
import sys
import torch
from torch.utils.data import Dataset, DataLoader



def download_dataset(dataset_id='D4RL/door/expert-v2'):
    dataset = minari.load_dataset(dataset_id, True)
    return dataset
    


def dataset_to_list(episodes):
    episodes = []
    for episode in dataset.iterate_episodes():
        observations = episode.observations
        next_observations = np.vstack([observations[1:], np.zeros((1, observations.shape[1]))])  # Add a row of zeros for the last next_observation
    
        episode_steps = []
        for i in range(len(episode.rewards)):
            step = {
                'observation': observations[i],
                'next_observation': next_observations[i],
                'reward': episode.rewards[i],
                'action': episode.actions[i],
                'termination': episode.terminations[i],
                'truncation': episode.truncations[i],
                'info': episode.infos['success'][i] if 'success' in episode.infos else None
                }   
            episode_steps.append(step)
        episodes.append(episode_steps)
    return episodes



def dataset_list_to_dataloder(episodes, bs=1, shuffle=True):
    class CustomDataset(Dataset):
        def __init__(self, episodes):
            self.episodes = episodes
            self.data = []
            for episode in episodes:
                self.data.extend(episode)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]

    custom_dataset = CustomDataset(episodes)
    dataloader = DataLoader(custom_dataset, batch_size=bs, shuffle=shuffle)
    return dataloader



