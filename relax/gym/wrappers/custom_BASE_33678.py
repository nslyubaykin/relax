import gym
import numpy as np


class ZerosMask(gym.Wrapper):  
    
    """
    Masks observation's array with zeros with `eps` probability.
    To simulate partial observability in a controlled manner.
    """
    
    def __init__(self, env, eps):
        
        gym.Wrapper.__init__(self, env)
        self.eps = eps
        
    def _mask_obs(self, obs):
        
        # create zero mask according to eps probability
        zero_mask = np.random.uniform(size=obs.shape) > self.eps
        
        # apply mask
        masked_obs = obs * zero_mask
        
        return masked_obs.copy()
    
    def step(self, action):
        
        # Step an upwrapped environment
        obs, reward, done, info = self.env.step(action)
        
        return self._mask_obs(obs), reward, done, info
    
    def reset(self):
        return self._mask_obs(self.env.reset())
    