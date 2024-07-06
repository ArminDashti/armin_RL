import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from . import utils



class PPO:
    def __init__(self, total_timesteps, env_id, n_env):
        self.vec_env = make_vec_env(env_id, n_envs=n_env)
    
    
    def learn(self, verbose, policy, total_timesteps, save_path):
        self.model = PPO(policy, self.vec_env, verbose=verbose)
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(save_path)
    
    
    def render(self):
        utils.render(self.model)



