import os
import torch
import numpy as np

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.distributions import Categorical, MultivariateNormal


def from_numpy(device, array, dtype=np.float32):
    array = array.astype(dtype)
    tensor = torch.from_numpy(array)
    return tensor.to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def detach_dist(dist):
    if isinstance(dist, MultivariateNormal):
        locs, cov = dist.loc.detach(), dist.covariance_matrix.detach()
        dist_detached = MultivariateNormal(loc=locs, covariance_matrix=cov)
    elif isinstance(dist, Categorical):
        probs = dist.probs.detach()
        dist_detached = Categorical(probs=probs)
    else:
        raise NotImplementedError(
            f'Distribution detach is not implemented for {type(dist).__name__}'
        )
    return dist_detached


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
        
        
class Checkpointer():
    
    def save_checkpoint(self, ckpt_dir, ckpt_name,
                        save_ckpt_attrs=True, verbose=True):
        
        # create directory if needed
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        checkpoint = {}

        # Save models if needed:
        if issubclass(type(self), nn.Module):
            checkpoint['models_state_dict'] = self.state_dict()
                

        # Special workaround for exploration
        if save_ckpt_attrs:
            if hasattr(self, 'exploration'):
                if hasattr(self.exploration, 'global_step'):
                    checkpoint['exploration_global_step'] = self.exploration.global_step

        # iterate over obj.__dict__:
        for attr_name, attr in self.__dict__.items():

            # saving optimizers & schedulers if any:
            if issubclass(type(attr), Optimizer) or issubclass(type(attr), _LRScheduler):
                checkpoint[attr_name] = attr.state_dict()

            # saving other specific attributes
            if save_ckpt_attrs:
                if hasattr(self, 'ckpt_attrs') and self.ckpt_attrs is not None:
                    if attr_name in self.ckpt_attrs:
                        checkpoint[attr_name] = attr

        torch.save(checkpoint, f'{ckpt_dir}/{ckpt_name}.pth')
        
        if verbose:
            print(f'Saved checkpoints for {type(self).__name__}...')
            print(*checkpoint.keys())

            
    def load_checkpoint(self, ckpt_dir, ckpt_name,
                        load_ckpt_attrs=True, verbose=True):
        
        checkpoint = torch.load(f'{ckpt_dir}/{ckpt_name}.pth')
        
        loaded_fields = []
        
        # loading models if needed:
        if issubclass(type(self), nn.Module):
            self.load_state_dict(checkpoint['models_state_dict'])
            loaded_fields.append('models_state_dict')
            del checkpoint['models_state_dict']
            
        # Special workaround for exploration
        if load_ckpt_attrs:
            if hasattr(self, 'exploration'):
                if hasattr(self.exploration, 'global_step') and 'exploration_global_step' in checkpoint.keys():
                    self.exploration.global_step = checkpoint['exploration_global_step']
                    loaded_fields.append('exploration_global_step')
                    del checkpoint['exploration_global_step']
        
        # iterate over obj.__dict__:
        for attr_name, attr in self.__dict__.items():

            # loading optimizers & schedulers if any:
            if issubclass(type(attr), Optimizer) or issubclass(type(attr), _LRScheduler):
                attr.load_state_dict(checkpoint[attr_name])
                loaded_fields.append(attr_name)

            # loading other specific attributes
            if load_ckpt_attrs:
                if hasattr(self, 'ckpt_attrs') and self.ckpt_attrs is not None:
                    if attr_name in self.ckpt_attrs:
                        setattr(self, attr_name, checkpoint[attr_name])
                        loaded_fields.append(attr_name)
                        
        if verbose:
            print(f'Loaded checkpoints for {type(self).__name__}...')
            print(*loaded_fields)
            