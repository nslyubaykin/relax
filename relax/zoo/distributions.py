from math import log, sqrt

import torch
from torch.distributions import Distribution, Normal
from torch.distributions.transforms import TanhTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution


class TanhUnivariateNormal(TransformedDistribution):
    
    def __init__(self, loc, log_scale,
                 acs_scale=1, acs_bias=0,
                 min_log_std=-20, max_log_std=2):

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.acs_scale = acs_scale
        self.acs_bias = acs_bias
        
        self.loc = loc
        
        self.log_scale = torch.clamp(
            log_scale,
            min=self.min_log_std,
            max=self.max_log_std
        )
        
        self.base_dist = Normal(
            loc=self.loc,
            scale=torch.exp(self.log_scale)
        )
        
        transforms = [TanhTransform(cache_size=1),
                      AffineTransform(cache_size=1, 
                                      loc=self.acs_bias,
                                      scale=self.acs_scale)]
        
        super().__init__(self.base_dist, transforms)
        
    @property
    def deterministic_acs(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu.detach()
    
    
class TanhNormal(Distribution):
    
    def __init__(self, loc, log_scale,
                 acs_scale=1, acs_bias=0,
                 min_log_std=-20, max_log_std=2):
        super(TanhNormal, self).__init__()
        
        self.base_dist = TanhUnivariateNormal(
            loc=loc, 
            log_scale=log_scale,
            acs_scale=acs_scale, 
            acs_bias=acs_bias,
            min_log_std=min_log_std, 
            max_log_std=max_log_std
        )
        
    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample()
    
    def log_prob(self, value):
        return self.base_dist.log_prob(value=value).sum(-1)
    
    @property
    def deterministic_acs(self):
        return self.base_dist.deterministic_acs
    
    @property
    def loc(self):
        return self.base_dist.loc
    
    @property
    def log_scale(self):
        return self.base_dist.log_scale
