# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:17:44 2024

@author: armin
"""

import gymnasium as gym
from gymnasium.utils.save_video import save_video
import sys


env = gym.make("FrozenLake-v1", render_mode="rgb_array_list")
_ = env.reset()
step_starting_index = 0
episode_index = 0
for step_index in range(199): 
   action = env.action_space.sample()
   _, _, terminated, truncated, _ = env.step(action)

   if terminated or truncated:
      save_video(env.render(), "videos", fps=env.metadata["render_fps"], step_starting_index=step_starting_index, episode_index=episode_index)
      step_starting_index = step_index + 1
      episode_index += 1
      env.reset()
      sys.exit()