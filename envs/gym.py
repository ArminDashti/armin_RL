import torch
import gymnasium
import numpy as np
from armin_utils import img, os, video, pytorch



class gym:
    def __init__(self, env_dir, video_format='mp4', seed=1234, device=None, game='HalfCheetah-v4', render_mode='rgb_array', max_step=None):
        if device is None:
            self.device = pytorch.detect_device()
        self.game = game
        self.render_mode = render_mode
        self.max_step = max_step
        self.env_dir = env_dir
        self.video_format = video_format
        self.step_num = 1
        self.episode_num = 1
        
        self._make(game, render_mode, max_step)
    
    @property
    def obs_dim(self):
        return self.env.observation_space

        
    
    @property
    def action_dim(self):
        return {'lowest_action':self.env.action_space.low, 
                'highest_action':self.env.action_space.high, 
                'range':(self.env.action_space.low[0], self.env.action_space.high[0]), 
                'shape':self.env.action_space.high.shape,
                'len':len(self.env.action_space.high)}
        
        
    def _make_dirs(self, env_dir, game_name):
        self.env_dir = os.make_dir([env_dir, game_name], True)
        self.frames_dir = os.make_dir([env_dir, game_name, 'frames'], True)
        self.videos_dir = os.make_dir([env_dir, game_name, 'videos'], True)
        self.logs_dir = os.make_dir([env_dir, game_name, 'logs'], True)
        
        
    def _make(self, game, render_mode, max_step):
        self.env = gymnasium.make(game, render_mode=render_mode)
        self._make_dirs(self.env_dir, game)
        if max_step is not None:
            self.env = gymnasium.wrappers.TimeLimit(self.env, max_episode_steps=max_step)
        self.reset()
    
    
    def reset(self):
        self.step_num = 1
        observation, reward = self.env.reset()
        result = {'observation':observation, 'reward':reward}
        return result
        
    
    def random_action(self):
        min_value = self.action_dim['range'][0]
        max_value = self.action_dim['range'][1]
        action_shape = self.action_dim['lowest_action'].shape
        random_array_constrained = np.random.uniform(min_value, max_value, size=action_shape)
        return random_array_constrained


    def step(self, action=None):
        if (not isinstance(action, np.ndarray)) and (action is not None):
            raise TypeError("Input must be a NumPy array, Use action_dim for more information")
        
        if action is None:
            action = self.random_action()

        observation, reward, terminated, truncated, info = self.env.step(action)
        result = {'observation':observation, 'reward':reward, 'terminated':terminated, 'truncated':truncated, 'info':info}
        self._save_frame(self.env.render())
        if (terminated == True) or (truncated == True):
            self._frames_to_video()
            self.reset()
        return result
    
    
    def _save_frame(self, array_img):
        img.save_img_in_dir(array_img, self.frames_dir, self.step_num, 'jpg')
        self.step_num += 1
    
    
    def _frames_to_video(self):
        video.frames_to_video(self.frames_dir, self.videos_dir, str(self.episode_num), self.video_format)
        self.episode_num += 1
    
#%%
env = gym('C:/users/armin/desktop/exp/')
env.step()