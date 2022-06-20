import torch

from torch import nn
from torch.nn import functional as F


class ContObsContAcsToObsMLP(nn.Module):
    
    def __init__(self, input_obs_dim, output_obs_dim,
                 acs_dim, nlayers,
                 nunits, activation=nn.Tanh(),
                 out_activation=nn.Identity()):
        
        super(ContObsContAcsToObsMLP, self).__init__()
        
        layers = []
        in_size = input_obs_dim + acs_dim
        for _ in range(nlayers):
            layers.append(nn.Linear(in_size, nunits))
            layers.append(activation)
            in_size = nunits
        
        self.layers = nn.Sequential(*layers)
        
        self.out_layer = nn.Sequential(
            nn.Linear(nunits, output_obs_dim),
            out_activation
        )

    def forward(self, obs, acs):
        
        out = self.layers(
            torch.cat(
                [torch.flatten(obs, start_dim=1), acs], 
                dim=-1
            )
        )
        out = self.out_layer(out)

        return out
    
    
class ContObsContAcsToRewsMLP(nn.Module):
    
    def __init__(self, input_obs_dim,
                 acs_dim, nlayers,
                 nunits, activation=nn.Tanh(),
                 out_activation=nn.Identity()):
        
        super(ContObsContAcsToRewsMLP, self).__init__()
        
        layers = []
        in_size = input_obs_dim + acs_dim
        for _ in range(nlayers):
            layers.append(nn.Linear(in_size, nunits))
            layers.append(activation)
            in_size = nunits
        
        self.layers = nn.Sequential(*layers)
        
        self.out_layer = nn.Sequential(
            nn.Linear(nunits, 1),
            out_activation
        )

    def forward(self, obs, acs):
        
        out = self.layers(
            torch.cat(
                [torch.flatten(obs, start_dim=1), acs], 
                dim=-1
            )
        )
        out = self.out_layer(out)

        return out
