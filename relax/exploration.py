import abc
import torch
import warnings
import numpy as np

from copy import deepcopy

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from relax.data.utils import *
from relax.torch.utils import *
from relax.data.sampling import PathList
from relax.schedules import init_schedule
from relax.data.replay_buffer import ReplayBuffer, BufferSample


class ActionAlterExploration():
    
    def reset_state(self):
        pass
        
    def save_state(self):
        return None
    
    def load_state(self, state):
        pass
    
    def get_logs(self):
        return {}


class EpsilonGreedy(ActionAlterExploration):
    
    def __init__(self, eps):
        
        self.eps = init_schedule(eps)
        self.global_step = 0
        
    def get_action(self, logits):
        
        if len(logits.shape) > 1:
            n_acs = logits.shape[0]
        else:
            n_acs = None
            
        eps_mask = np.random.random(n_acs) < self.eps.value(self.global_step)
        
        random_acs = np.random.random(logits.shape).argmax(-1)
        critic_acs = logits.argmax(-1)
        
        out_acs = np.where(eps_mask, random_acs, critic_acs)
        
        if n_acs is None:
            out_acs = int(out_acs)
            
        return out_acs
    
    def schedules_step(self):
        self.global_step += 1
        
    def get_logs(self):
        logs = {}
        pr = type(self).__name__
        logs[f'{pr}_eps'] = self.eps.value(self.global_step)
        logs[f'{pr}_global_step'] = self.global_step
        return logs

    
class OrnsteinUhlenbeck(ActionAlterExploration):
    
    def __init__(self, 
                 theta,
                 sigma, 
                 dim, 
                 mu=0., 
                 dt=1e-2, 
                 x0=None,
                 n_random_steps=None,
                 min_acs=None,
                 max_acs=None):
        
        self.theta = init_schedule(theta)
        self.sigma = init_schedule(sigma)
        
        self.dt = dt
        self.mu = mu
        
        self.x0 = x0
        self.dim = dim
        
        self.reset_state()
        assert self.x_prev.shape[0] == self.dim
        
        self.global_step = 0
        self.counter = 0
        self.n_random_steps = n_random_steps
        
        self.min_acs = min_acs
        self.max_acs = max_acs
        
    def sample(self):
        x = self.x_prev \
            + self.theta.value(self.global_step) * (self.mu - self.x_prev) * self.dt \
            + self.sigma.value(self.global_step) * np.sqrt(self.dt) * np.random.normal(size=self.dim)
        self.x_prev = x
        return x
    
    def reset_state(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.dim)
        
    def save_state(self):
        return self.x_prev
    
    def load_state(self, state):
        if state is not None:
            self.x_prev = state
            
    def get_action(self, acs: np.ndarray) -> np.ndarray:
        rand_acs = float(self.n_random_steps is not None and self.global_step <= self.n_random_steps)
        out_acs = acs * (1-rand_acs) + self.sample()
        if self.min_acs is not None or self.max_acs is not None:
            out_acs = np.clip(out_acs, a_min=self.min_acs, a_max=self.max_acs)
        return out_acs
    
    def get_logs(self):
        logs = {}
        pr = type(self).__name__
        logs[f'{pr}_sigma'] = self.sigma.value(self.global_step)
        logs[f'{pr}_theta'] = self.theta.value(self.global_step)
        logs[f'{pr}_global_step'] = self.global_step
        return logs
        
    def schedules_step(self):
        self.global_step += 1
        
        
class RandomNormal(ActionAlterExploration):
    
    def __init__(self,
                 sigma, 
                 mu=0.,
                 n_random_steps=None,
                 min_acs=None,
                 max_acs=None):
        
        self.sigma = init_schedule(sigma)
        self.mu = init_schedule(mu)
        
        self.global_step = 0
        self.n_random_steps = n_random_steps
        
        self.min_acs = min_acs
        self.max_acs = max_acs
        
    def schedules_step(self):
        self.global_step += 1
        
    def get_action(self, acs: np.ndarray) -> np.ndarray:
        sigma = self.sigma.value(self.global_step)
        mu = self.mu.value(self.global_step)
        noise = np.random.normal(mu, sigma, acs.shape)
        rand_acs = float(self.n_random_steps is not None and self.global_step <= self.n_random_steps)
        out_acs = acs * (1-rand_acs) + noise
        if self.min_acs is not None or self.max_acs is not None:
            out_acs = np.clip(out_acs, a_min=self.min_acs, a_max=self.max_acs)
        return out_acs
    
    def get_logs(self):
        logs = {}
        pr = type(self).__name__
        logs[f'{pr}_sigma'] = self.sigma.value(self.global_step)
        logs[f'{pr}_global_step'] = self.global_step
        return logs

    
class RND(Checkpointer,
          nn.Module):
    
    def __init__(self,
                 random_net,
                 distilling_net,
                 learning_rate,
                 device,
                 batch_size,
                 n_steps_per_update=1,
                 update_freq=1,
                 weight_decay=0.0,
                 obs_norm_clip=5,
                 # reward normalization history
                 int_rews_history=100000,
                 # obs normalization for off-policy algs
                 stats_recalc_freq=1,
                 buffer_stats_sample=25000,
                 # obs normalization for on policy algs
                 obs_stats_history=100000,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # initialize all scheduled args as schedules
        # if they are already not schedules
        self.global_step = 0 
        self.local_step = 0
        self.n_updates = 0
        self.n_stats_updates = 0
        self.n_model_resets = 0
        self.ckpt_attrs = ['global_step', 'local_step', 'n_updates', 'n_stats_updates',
                           'buffer_stats', 'int_rews_std']
        self.obs_norm_clip = obs_norm_clip
        
        self.learning_rate = init_schedule(learning_rate)
        self.batch_size = init_schedule(batch_size, discrete=True)
        self.update_freq = init_schedule(update_freq, discrete=True)
        self.stats_recalc_freq = init_schedule(stats_recalc_freq, discrete=True)
        self.n_steps_per_update = init_schedule(n_steps_per_update, discrete=True)
        self.stats_recalc_freq = init_schedule(stats_recalc_freq, discrete=True)
        
        # nn.Module params  
        self.device = device
        
        # distilling net
        self.distilling_net = distilling_net
        self.distilling_net.to(self.device)
        
        self.optimizer = optim.Adam(self.distilling_net.parameters(), 
                                    lr=1, 
                                    eps=1e-6,
                                    weight_decay=weight_decay) 
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lambda t: self.learning_rate.value(t))
        
        # random net
        self.random_net = random_net
        self.random_net.to(self.device)
        
        # make random net untrainable
        for p in self.random_net.parameters():
            p.requires_grad = False
        
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        # loging:
        self.last_logs = {}
        
        # Buffer stats section 
        self.buffer_stats = {}
        self.buffer_stats_sample = buffer_stats_sample
        
        # Obs stats section (on-policy case)
        self.obs_buffer = None
        self.obs_stats_history = obs_stats_history
        
        # Intristic rewards section
        self.int_rews_history = int_rews_history
        self.int_rews_buffer = []
        self.int_rews_mean = None
        self.int_rews_std = None
        
    def schedules_step(self):
        self.global_step += 1
        if hasattr(self, 'scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
                
    def update_int_rew_stats(self):
        self.int_rews_std = np.std(self.int_rews_buffer)
        self.int_rews_mean = np.mean(self.int_rews_buffer)
            
    def update_buffer_stats(self, data: (ReplayBuffer, PathList)):
        
        if not isinstance(data, ReplayBuffer):
            
            # PathList case
            # create observation history buffer if needed
            if self.obs_buffer is None:
                self.obs_buffer = ReplayBuffer(self.obs_stats_history)

            # add new pathlist to the buffer
            self.obs_buffer.add_paths(data)
            
            buffer = self.obs_buffer
            
        else:
            
            # ReplayBuffer case
            buffer = data
        
        if self.buffer_stats_sample is not None:
            # Calculate stats using the subset of buffer 
            stats_data = buffer.sample(
                min(buffer.n_transitions, self.buffer_stats_sample)  
            )
        else:
            # Calculate stats using the entire buffer 
            stats_data = buffer
        
        lag_obs = handle_lags(data=stats_data,
                              fields={'obs': 'lag_concat_obs'},
                              nlags=self.obs_nlags,
                              concat_axis=self.obs_concat_axis,
                              expand_axis=self.obs_expand_axis,
                              padding=self.obs_padding)
        stats_data.drop_field('lag_concat_obs')

        self.buffer_stats = {
            'lag_obs_mean': np.mean(lag_obs, axis=0),
            'lag_obs_std': np.std(lag_obs, axis=0),
        }
            
        self.n_stats_updates += 1
        
    def forward(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        
        # estimate random net targets:
        with torch.no_grad():
            random_target = self.random_net(obs)
            
        # estimate distilling net outputs
        distillation = self.distilling_net(obs)
        
        # calculate error for each obs
        pred_error = F.mse_loss(
            distillation, 
            random_target,
            reduction='none'
        ).mean(axis=-1)
        
        return pred_error
    
    def forward_np(self, obs):
        raise NotImplementedError
    
    def estimate_novelty(self, data: (PathList, BufferSample)) -> torch.FloatTensor:
        
        # preprocess observations
        lag_obs_norm = self._prepare_obs(data=data,
                                         next_obs=True)
        
        batch_size = self.batch_size.value(self.global_step)
        
        if batch_size >= lag_obs_norm.shape[0]:
            
            # single forward pass
            # get discillation error and detach it
            pred_error = self.forward(lag_obs_norm).detach()
            
        else:
            
            # predict in batches
            pred_error = []
            
            dataloader = DataLoader(
                lag_obs_norm, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            for lag_obs_norm_i in dataloader:
                
                pred_error_i = self.forward(lag_obs_norm_i).detach()
                
                pred_error.append(pred_error_i)
                
            pred_error = torch.cat(pred_error, dim=0)
            
        # Standardize intristic rewards
        # if first global step - just use current stats for normalization
        # otherwise use pre-clalculated stats
        if self.global_step == 0 and self.int_rews_std is None:
            #pred_error -= pred_error.mean() # Changed
            pred_error /= pred_error.std()
        else:
            #pred_error -= self.int_rews_mean # Changed
            pred_error /= self.int_rews_std
            
        return pred_error
        
    def update(self, data: (PathList, ReplayBuffer)) -> dict:
        
        if not isinstance(data, ReplayBuffer):
            
            # PathList case
            logs = self._on_policy_update(pathlist=data)
            
        else:
            
            # ReplayBuffer case
            logs = self._off_policy_update(buffer=data)
            
        return logs
    
    def _prepare_obs(self, 
                     data: (PathList, 
                            ReplayBuffer, 
                            BufferSample),
                     next_obs=False) -> torch.FloatTensor:
        
        # creating lags if needed in model
        # unpack rollouts for training
        
        field = 'obs'
        if next_obs:
            field = 'next_obs'
            data.add_next_obs()
        
        lag_obs = handle_lags(data=data,
                              fields={field: 'lag_concat_obs'},
                              nlags=self.obs_nlags,
                              concat_axis=self.obs_concat_axis,
                              expand_axis=self.obs_expand_axis,
                              padding=self.obs_padding)
        
        data.drop_field('lag_concat_obs')
        
        if next_obs:
            data.drop_field('next_obs')
        
        # normalize observations for training
        # if first global step - just use current stats for normalization
        # otherwise use pre-clalculated stats
        if self.global_step == 0 and len(self.buffer_stats) == 0:
            lag_obs_norm = normalize(data=lag_obs,
                                     mean=np.mean(lag_obs, axis=0),
                                     std=np.std(lag_obs, axis=0))
        else:
            lag_obs_norm = normalize(data=lag_obs,
                                     mean=self.buffer_stats['lag_obs_mean'],
                                     std=self.buffer_stats['lag_obs_std'])

        # clip obs
        clip_val = abs(self.obs_norm_clip)
        lag_obs_norm = np.clip(a=lag_obs_norm,
                               a_min=-clip_val, 
                               a_max=clip_val)
        
        # convert to tensor
        lag_obs_norm = from_numpy(self.device, lag_obs_norm)
        
        return lag_obs_norm
        
    def _on_policy_update(self, pathlist: PathList) -> dict:
        
        # update buffer stats 
        self.update_buffer_stats(data=pathlist)
        
        # preprocess observations
        lag_obs_norm = self._prepare_obs(data=pathlist)
        
        # create dataloader
        batch_size = self.batch_size.value(self.global_step)
        
        dataloader = DataLoader(
            lag_obs_norm, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # perform training loop
        n_steps = self.n_steps_per_update.value(self.global_step)
        update_freq = self.update_freq.value(self.global_step)
    
        # calculate the index of the last step
        last_step = n_steps - 1
        
        for step in range(n_steps):
                
            # iterate through the dataloader
            mean_loss = []
            pred_error_log = []

            for lag_obs_norm_i in dataloader:

                # estimate prediction errors 
                pred_error = self.forward(lag_obs_norm_i)

                # Add iterative prediction error
                if step == last_step: # done only during the last epoch
                    pred_error_log_i = list(to_numpy(pred_error))
                    pred_error_log.extend(pred_error_log_i)

                # formulate discillation loss
                loss = torch.mean(pred_error)

                # performing gradient step:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mean_loss.append(loss.item())

            # add prediction error to int rews buffer
            if step == last_step: # done only during the last epoch
                self.int_rews_buffer.extend(pred_error_log)
                self.int_rews_buffer = self.int_rews_buffer[-self.int_rews_history:]

            self.n_updates += 1
            
            self.local_step += 1

            # logging
            pr = type(self).__name__
            self.last_logs = {f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                              f'{pr}_loss': np.mean(mean_loss),
                              f'{pr}_batch_size': batch_size,
                              f'{pr}_global_step': self.global_step,
                              f'{pr}_local_step': self.local_step,
                              f'{pr}_n_updates': self.n_updates,
                              f'{pr}_n_stats_updates': self.n_stats_updates}

        # update intristic rewards stats
        self.update_int_rew_stats()
        
        # global step for schedules
        self.schedules_step()
        
        return self.last_logs
    
    def _off_policy_update(self, buffer: ReplayBuffer) -> dict:
        
        # perform training loop
        for _ in range(self.n_steps_per_update.value(self.global_step)):
            
            # update buffer stats is needed
            if (len(self.buffer_stats.keys()) == 0
                or self.local_step % self.stats_recalc_freq.value(self.global_step) == 0):
                
                self.update_buffer_stats(data=buffer)
                
            if (self.learning_rate.value(self.global_step) > 0
                and self.local_step % self.update_freq.value(self.global_step) == 0):
                
                # sampling self.batch_size transitions:
                batch_size = self.batch_size.value(self.global_step)
                
                # sample
                sample = buffer.sample(batch_size=batch_size)
                
                # preprocess observations
                lag_obs_norm = self._prepare_obs(data=sample)
                
                # estimate prediction errors 
                pred_error = self.forward(lag_obs_norm)
                
                # add prediction error to int rews buffer
                pred_error_log = list(to_numpy(pred_error))
                self.int_rews_buffer.extend(pred_error_log)
                self.int_rews_buffer = self.int_rews_buffer[-self.int_rews_history:]
                
                # formulate discillation loss
                loss = torch.mean(pred_error)
                
                # performing gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                    
                self.n_updates += 1
                
                # logging
                pr = type(self).__name__
                self.last_logs = {f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                                  f'{pr}_loss': loss.item(),
                                  f'{pr}_batch_size': batch_size,
                                  f'{pr}_global_step': self.global_step,
                                  f'{pr}_local_step': self.local_step,
                                  f'{pr}_n_updates': self.n_updates,
                                  f'{pr}_n_stats_updates': self.n_stats_updates}
                
            # update intristic rewards stats if needed
            if (self.int_rews_std is None
                or self.local_step % self.stats_recalc_freq.value(self.global_step) == 0):
                
                if len(self.int_rews_buffer) > 0:
                    self.update_int_rew_stats()
            
            self.local_step += 1
        
        # global step for schedules
        self.schedules_step()
        
        return self.last_logs
