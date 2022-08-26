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
        
        # Step an unwrapped environment
        obs, reward, done, info = self.env.step(action)
        
        return self._mask_obs(obs), reward, done, info
    
    def reset(self):
        return self._mask_obs(self.env.reset())
    
    
class NormalNoise(gym.Wrapper):
    
    """
    Adds normal noise with `scale` std and `loc` mean to an observation array
    To simulate partial observability in a controlled manner.
    """
    
    def __init__(self, env, scale, loc=0):
        
        gym.Wrapper.__init__(self, env)
        self.scale = scale
        self.loc = loc
        
    def _add_noise(self, obs):
        
        # sample noise according to `scale` and `loc`
        noise = np.random.normal(scale=self.scale,
                                 loc=self.loc,
                                 size=obs.shape)
        
        # apply mask
        noisy_obs = obs + noise
        
        return noisy_obs.copy()
    
    def step(self, action):
        
        # Step an unwrapped environment
        obs, reward, done, info = self.env.step(action)
        
        return self._add_noise(obs), reward, done, info
    
    def reset(self):
        return self._add_noise(self.env.reset())
    
