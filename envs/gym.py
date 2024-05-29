import gymnasium as gym
import os
from PIL import Image
from armin_utils.utils.to_video import imgs_to_video
from armin_utils.utils import files
from armin_utils.utils.tensor import tensor_to_numpy, to_tensor
import torch


class gym:
    def __init__(self, env_dir, 
                 to_tensor=['action'], 
                 render_mode='rgb_array', 
                 game='HalfCheetah-v4', 
                 max_episode_steps=None, 
                 video_format='.mp4',
                 seed=1234,
                 device=None,
                 ):
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.to_tensor = to_tensor
        print(game)
        self.game = game
        self.render_mode = render_mode
        self.env_dir = env_dir
        self.video_format = video_format
        self.max_episode_steps = max_episode_steps
        self.temp_shots_dir = os.path.join(env_dir, 'temp_shots')
        self.videos_dir = os.path.join(env_dir, 'videos')
        self.make_dirs()
        self.shot_idx = 1
        self.eps_idx = 1
        self.make()
        
        
    def make_dirs(self):
        if not os.path.exists(self.temp_shots_dir):
            os.makedirs(self.temp_shots_dir)
        else:
            files.remove_files(self.temp_shots_dir)
        
        if not os.path.exists(self.videos_dir):
            os.makedirs(self.videos_dir)
        else:
            files.remove_files(self.videos_dir)
    
        
    def make(self):
        
        self.env = gym.make(self.game, render_mode=self.render_mode)
        if self.max_episode_steps is not None:
            self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=self.max_episode_steps)
        self.reset()
        self.action_space = self.env.action_space
    
    
    def reset(self):
        self.shot_idx = 1
        observation, reward = self.env.reset()
        result = {'observation':observation, 'reward':reward}
        return result
        
    
    def step(self, action):
        if torch.is_tensor(action):
            action = tensor_to_numpy(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        result = {'observation':observation, 'reward':reward, 'terminated':terminated, 'truncated':truncated, 'info':info}
        self.save_shot()
        if (terminated == True) or (truncated == True):
            self.to_video()
        return result
    
    
    def save_shot(self):
        RGB = self.env.render()
        RGB = Image.fromarray(RGB, 'RGB')
        RGB_save_dir = os.path.join(self.env_dir, 'temp_shots')
        RGB.save(RGB_save_dir+'/'+str(self.shot_idx)+'.jpg')
        self.shot_idx += 1
    
    
    def to_video(self):
        imgs_to_video(self.temp_shots_dir, self.videos_dir, 'video_output.avi')
        self.eps_idx += 1
    
    
        
