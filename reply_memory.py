import random
import numpy as np
from collections import namedtuple
import torch

# https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/replay_memory.py
class reply_memory:
    def __init__(self, capacity, seed=123):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0


    def push(self, observations, actions, rewards, next_observations, truncations):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (observations, actions, rewards, next_observations, truncations)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size, tensor=False):
        batch = random.sample(self.buffer, batch_size)
        observations, actions, rewards, next_observations, truncations = map(np.stack, zip(*batch))
        if tensor == False:
            return {'observations':observations, 'actions':actions, 'rewards':rewards, 'next_observations':next_observations, 'truncations':truncations}
        else:
            return {'observations':torch.tensor(observations).float(), 
                    'actions':torch.tensor(actions).float(), 
                    'rewards':torch.tensor(rewards).float(), 
                    'next_observations':torch.tensor(next_observations).float(), 
                    'truncations':torch.tensor(truncations).float()}


    def __len__(self):
        return len(self.buffer)


    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))
        
        
        
# Taken from https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
Transition = namedtuple('Transition',('state', 'action', 'done', 'next_state', 'reward'))

class reply_memory_tuple:
    def __init__(self, capacity, saving_tensor=True):
        self.saving_tensor = saving_tensor
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)