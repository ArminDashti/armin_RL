# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 22:44:31 2024

@author: armin
"""

class Base:
    def observation_dim(self):
        return self.dataset[0].observations.shape[1:]
    
    
    def action_dim(self):
        return self.dataset[0].actions.shape[1:]
    
    
    @property
    def episodes_num(self):
        return (sum(1 for _ in self.dataset.iterate_episodes()))
    
    
    @property
    def time_steps_num(self):
        return self.dataset[0].total_timesteps