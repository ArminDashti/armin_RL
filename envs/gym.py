import torch
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

import numpy as np
from armin_utils import dir_utils
from armin_pytorch.device import detect_device
import os


class Make:
    def __init__(self, 
                 env_dir, 
                 seed=1234, 
                 device='auto', 
                 env_name='MountainCarContinuous-v0', 
                 render_mode='rgb_array_list', 
                 max_step=None):
        
        if device == 'auto':
            self.device = detect_device()
        
        self.env = self._make_env(env_name, render_mode=render_mode, max_step=max_step)
        self._make_dirs(env_dir)
        self.space_type = self.identify_space_type(self.env)
        
        
    def _make_env(self, env_name, render_mode, max_step):
        env = gym.make(env_name, render_mode=render_mode)
        if max_step is not None:
            env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_step)
        env.reset()
        return env
        
        
    def identify_space_type(self, env):
        if isinstance(env.action_space, Discrete):
            return 'discrete'
        else:
            return 'continuous'
        
        
    def _make_dirs(self, env_dir):
        saved_video_dir = os.path.join(env_dir, 'saved_videos')
        dir_utils.make_dir(saved_video_dir, force=True)
    
    
    def observation_space(self, only_dim=True):
        if only_dim == True:
            return self.env.observation_space.n
        else:
            return self.env.observation_space

    
    def _discrete_action_space(self, only_dim=True):
        if only_dim == True:
            return self.env.observation_space.n
        else:
            return self.env.action_space
    
    
    def _continuous_action_space(self):
        return {'lowest_action':self.env.action_space.low, 
                'highest_action':self.env.action_space.high, 
                'range':(self.env.action_space.low[0], self.env.action_space.high[0]), 
                'shape':self.env.action_space.high.shape,
                'len':len(self.env.action_space.high)}


    def action_space(self, only_dim=True):
        if self.space_type == 'discrete':
            return self._discrete_action_space(only_dim=only_dim)
        else:
            return self._continuous_action_space()
    
    
    def reset(self):
        observation, reward = self.env.reset()
        result = {'observation':observation, 'reward':reward}
        return result
    
    
    def _discrete_random_action(self):
        pass
    
    
    def _continuous_random_action(self):
        action_space = self._continuous_action_space()
        min_value = action_space['range'][0]
        max_value = action_space['range'][1]
        action_shape = action_space['lowest_action'].shape
        random_array = np.random.uniform(min_value, max_value, size=action_shape)
        return random_array
    
    
    def random_action(self):
        if self.space_type == 'discrete':
            pass
        else:
            random_action = self._continuous_random_action()
            return random_action
        
        
    def step(self, action=None):
        if action is None:
            action = self.random_action()
            
        if not isinstance(action, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        observation, reward, terminated, truncated, info = self.env.step(action)
        result = {'observation':observation, 'reward':reward, 'terminated':terminated, 'truncated':truncated, 'info':info}
        if (terminated == True) or (truncated == True):
            self.reset()
        return result



        
    
        
        

env = Make('C:/users/armin/desktop/exp/')
env.step()