import abc
import torch
import warnings
import numpy as np

from warnings import warn
from copy import deepcopy

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from relax.data.utils import normalize, unnormalize
from relax.data.utils import handle_lags, handle_n_step
from relax.data.sampling import PathList
from relax.data.replay_buffer import ReplayBuffer, BufferSample
from relax.data.acceleration import DynaAccelerator
from relax.zoo.layers import NoisyLinear
from relax.schedules import init_schedule
from relax.torch.utils import *


class BaseCritic(nn.Module, Checkpointer, metaclass=abc.ABCMeta):
    
    def __init__(self, 
                 critic_net: nn.Module,
                 device,
                 learning_rate,
                 gamma,
                 n_target_updates=1, 
                 n_steps_per_update=1,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first'): 
        super().__init__()
        
        # initialize counters
        self.global_step = 0
        self.n_updates = 0
        self.ckpt_attrs = ['global_step', 'n_updates']
        
        # initialize schedules
        self.gamma = init_schedule(gamma)
        self.learning_rate = init_schedule(learning_rate)
        self.n_target_updates = init_schedule(n_target_updates, discrete=True)
        self.n_steps_per_update = init_schedule(n_steps_per_update, discrete=True)
            
        # initialize torch objects
        self.device = device
        self.critic_net = critic_net
        self.optimizer = optim.Adam(self.critic_net.parameters(), lr=1, eps=1e-6)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lambda t: self.learning_rate.value(t))
        self.critic_net.to(self.device)
        
        # initialize constant params
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        # other params
        self.last_logs = {}
        
    def forward(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        return self.critic_net(obs)
    
    def forward_np(self, obs: np.ndarray) -> np.ndarray:
        obs = from_numpy(self.device, obs)
        return to_numpy(self.forward(obs))
    
    def schedules_step(self):
        self.global_step += 1
        if hasattr(self, 'scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
            
    def set_device(self, device):
        self.device = device
        self.critic_net.to(self.device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())
    
    def update(self, paths: PathList) -> dict:
        raise NotImplementedError
        
    def estimate_value(self, paths: PathList) -> np.ndarray:
        raise NotImplementedError
        
    def estimate_qvalue(self, paths: PathList) -> np.ndarray:
        raise NotImplementedError
        
    def estimate_advantage(self, paths: PathList) -> np.ndarray:
        raise NotImplementedError


class Baseline(BaseCritic):
    
    def __init__(self,
                 critic_net: nn.Module,
                 device,
                 learning_rate,
                 batch_size,
                 gamma,
                 n_target_updates=1, 
                 n_steps_per_update=1,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first'):
        
        super().__init__(critic_net,
                         device,
                         learning_rate,
                         gamma,
                         n_target_updates, 
                         n_steps_per_update,
                         obs_nlags,
                         obs_concat_axis,
                         obs_expand_axis,
                         obs_padding)
        
        self.batch_size = init_schedule(batch_size, discrete=True)
    
    def update(self, paths: PathList) -> dict:
        
        # creating lags if needed in model
        # unpack rollouts for training
        obs = handle_lags(data=paths,
                          fields={'obs': 'lag_concat_obs'},
                          nlags=self.obs_nlags,
                          concat_axis=self.obs_concat_axis,
                          expand_axis=self.obs_expand_axis,
                          padding=self.obs_padding)
        
        # estimate Q-values if needed
        if 'rews_to_go' not in paths.rollouts[0].data.keys():
            paths.add_disc_cumsum('rews_to_go', 'rews',
                                  self.gamma.value(self.global_step))
        
        # unpack rollouts for training
        q_values = paths.unpack(['rews_to_go'])
        
        if self.n_target_updates.value(self.global_step) > 1:
            
            warn(f'For {type(self).__name__} n_target_updates greater than 1 is invalid, setting back to 1',
                 UserWarning)
            
            self.n_target_updates = init_schedule(1, discrete=True)
        
        for _ in range(self.n_target_updates.value(self.global_step)):
        
            # normalize Q values and convert them to tensor
            targets = normalize(q_values, q_values.mean(), q_values.std())
            obs, targets = from_numpy(self.device, obs), from_numpy(self.device, targets)
            
            dataloader = DataLoader(
                list(zip(obs,
                         targets)), 
                batch_size=self.batch_size.value(self.global_step),
                shuffle=True,
                num_workers=0
            )

            for _ in range(self.n_steps_per_update.value(self.global_step)):
                
                mean_loss = []
                
                for obs_i, targets_i in dataloader:

                    # obtain baselines fit
                    baseline_fit = torch.squeeze(self.forward(obs_i))

                    # check if the shapes are right:
                    assert baseline_fit.shape == targets_i.shape

                    # calculate critic loss
                    loss = F.mse_loss(baseline_fit, targets_i)

                    # performing gradient step:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    mean_loss.append(loss.item())
                
                self.n_updates += 1
        
        pr = type(self).__name__
        self.last_logs = {f'{pr}_critic_loss':  np.mean(mean_loss),
                          f'{pr}_global_step': self.global_step,
                          f'{pr}_n_updates': self.n_updates,
                          f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                          f'{pr}_gamma': self.gamma.value(self.global_step),
                          f'{pr}_n_steps_per_update': self.n_steps_per_update.value(self.global_step)}
        
        self.schedules_step()
        
        return self.last_logs
            
    def estimate_advantage(self, paths: PathList) -> np.ndarray:
        
        # creating lags if needed in model
        # unpack rollouts for training
        obs = handle_lags(data=paths,
                          fields={'obs': 'lag_concat_obs'},
                          nlags=self.obs_nlags,
                          concat_axis=self.obs_concat_axis,
                          expand_axis=self.obs_expand_axis,
                          padding=self.obs_padding)
        
        # estimate Q-values if needed
        if 'rews_to_go' not in paths.rollouts[0].data.keys():
                paths.add_disc_cumsum('rews_to_go', 'rews',
                                      self.gamma.value(self.global_step))
        
        # unpack rollouts for training
        q_values = paths.unpack(['rews_to_go'])
        
        # predict baselines with critic net 
        baselines_normalized = self.forward_np(obs).squeeze()
        
        # check if dimensions are OK
        assert baselines_normalized.shape == q_values.shape
        
        # unnormalize baselines to match Q values distribution
        baselines = unnormalize(baselines_normalized, q_values.mean(), q_values.std())
        
        # compute advantages
        advantages = q_values - baselines
        
        return advantages
    
    
class GAE(Baseline):
    
    def __init__(self, 
                 critic_net: nn.Module,
                 device,
                 learning_rate,
                 batch_size,
                 gamma,
                 gae_lambda,
                 n_target_updates=1, 
                 n_steps_per_update=20,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first'):
        
        super().__init__(critic_net,
                         device,
                         learning_rate,
                         batch_size,
                         gamma,
                         n_target_updates, 
                         n_steps_per_update,
                         obs_nlags,
                         obs_concat_axis,
                         obs_expand_axis,
                         obs_padding)
        
        self.gae_lambda = init_schedule(gae_lambda)
    
    def estimate_advantage(self, paths: PathList) -> np.ndarray:
        
        # querying 'next_obs' field:
        paths.add_next_obs()
        
        # creating lags if needed in model
        # unpack rollouts for training
        obs, next_obs = handle_lags(data=paths,
                                    fields={'obs': 'lag_concat_obs',
                                            'next_obs': 'lag_concat_next_obs'},
                                    nlags=self.obs_nlags,
                                    concat_axis=self.obs_concat_axis,
                                    expand_axis=self.obs_expand_axis,
                                    padding=self.obs_padding)
        
        # estimate Q-values if needed
        if 'rews_to_go' not in paths.rollouts[0].data.keys():
            paths.add_disc_cumsum('rews_to_go', 'rews',
                                  self.gamma.value(self.global_step))
        
        # unpack rollouts for training
        rews, q_values = paths.unpack(['rews', 'rews_to_go'])   
        
        # predict baselines with critic net 
        baselines_normalized = self.forward_np(obs).squeeze()
        # unnormalize baselines to match Q values distribution
        baselines = unnormalize(baselines_normalized, q_values.mean(), q_values.std())
        
        # predict baselines t+1 with critic net 
        baselines_normalized_next = self.forward_np(next_obs).squeeze()
        # unnormalize baselines to match Q values distribution
        baselines_next = unnormalize(baselines_normalized_next, q_values.mean(), q_values.std())
        
        # check if dimensions are OK
        assert baselines.shape == rews.shape and baselines_next.shape == rews.shape
        
        # estimate GAE lambda deltas
        deltas = rews + self.gamma.value(self.global_step) * baselines_next - baselines
        
        # pack deltas to paths:
        paths.pack(deltas.tolist(), 'deltas')
        
        # estimate gamma and lambda discounted deltas
        paths.add_disc_cumsum('deltas_to_go', 'deltas',
                              self.gamma.value(self.global_step) * self.gae_lambda.value(self.global_step))
        
        advantages = paths.unpack(['deltas_to_go'])
        
        # delete unneeded data fields
        for rm_field in ['deltas_to_go', 'deltas']:
            paths.drop_field(rm_field)
            
        return advantages


class BootstrappedContinuous(BaseCritic):
    
    def __init__(self, 
                 critic_net: nn.Module,
                 device,
                 learning_rate,
                 gamma,
                 n_target_updates=5, 
                 n_steps_per_update=20,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first'): 
        super().__init__(critic_net,
                         device,
                         learning_rate,
                         gamma,
                         n_target_updates, 
                         n_steps_per_update,
                         obs_nlags,
                         obs_concat_axis,
                         obs_expand_axis,
                         obs_padding)
    
    def update(self, paths: PathList) -> dict:
        
        # querying 'next_obs' field:
        paths.add_next_obs()
            
        # creating lags if needed in model
        # unpack rollouts for training
        obs, next_obs = handle_lags(data=paths,
                                    fields={'obs': 'lag_concat_obs',
                                            'next_obs': 'lag_concat_next_obs'},
                                    nlags=self.obs_nlags,
                                    concat_axis=self.obs_concat_axis,
                                    expand_axis=self.obs_expand_axis,
                                    padding=self.obs_padding)    
        
        acs, rews, terminals = paths.unpack(['acs', 'rews', 'terminals'])
        
        obs = from_numpy(self.device, obs)
        acs = from_numpy(self.device, acs)
        next_obs = from_numpy(self.device, next_obs)
        rews = from_numpy(self.device, rews)
        terminals = from_numpy(self.device, terminals)
        
        for _ in range(self.n_target_updates.value(self.global_step)):
            
            # calculate targets using Vt+1
            Vsnp1 = torch.squeeze(self.forward(next_obs))
            targets = rews + self.gamma.value(self.global_step) * Vsnp1 * torch.logical_not(terminals)
            targets = targets.detach()
            
            for _ in range(self.n_steps_per_update.value(self.global_step)):
                
                # obtain targets_fit using Vt
                targets_fit = torch.squeeze(self.forward(obs))
                
                # check if the shapes are right:
                assert targets_fit.shape == targets.shape

                # calculate critic loss
                loss = F.mse_loss(targets_fit, targets)

                # performing gradient step:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.n_updates += 1
        
        pr = type(self).__name__
        self.last_logs = {f'{pr}_critic_loss':  loss.item(),
                          f'{pr}_global_step': self.global_step,
                          f'{pr}_n_updates': self.n_updates,
                          f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                          f'{pr}_gamma': self.gamma.value(self.global_step),
                          f'{pr}_n_steps_per_update': self.n_steps_per_update.value(self.global_step)}
        
        self.schedules_step()
        
        return self.last_logs

    def estimate_advantage(self, paths: PathList) -> np.ndarray:
        
        # querying 'next_obs' field:
        paths.add_next_obs()
        
        # creating lags if needed in model
        # unpack rollouts for training
        obs, next_obs = handle_lags(data=paths,
                                    fields={'obs': 'lag_concat_obs',
                                            'next_obs': 'lag_concat_next_obs'},
                                    nlags=self.obs_nlags,
                                    concat_axis=self.obs_concat_axis,
                                    expand_axis=self.obs_expand_axis,
                                    padding=self.obs_padding)     
        
        rews, terminals = paths.unpack(['rews', 'terminals'])
        
        # query the critic to get Vt
        Vsn = self.forward_np(obs).squeeze()
        
        # query the critic to get Vt+1
        Vsnp1 = self.forward_np(next_obs).squeeze()
        
        # check if dimensions are OK
        assert Vsn.shape == rews.shape and Vsnp1.shape == rews.shape
        
        # estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        Qsan = rews + self.gamma.value(self.global_step) * Vsnp1 * np.logical_not(terminals)
        
        # compute advantages
        advantages = Qsan - Vsn
        
        return advantages
    

class DQN(nn.Module,
          DynaAccelerator,
          Checkpointer, 
          metaclass=abc.ABCMeta):
    
    def __init__(self, 
                 critic_net: nn.Module,
                 device,
                 learning_rate,
                 batch_size,
                 gamma=0.99,
                 cql_alpha=0.0,
                 grad_norm_clipping=10,
                 td_criterion=F.smooth_l1_loss,
                 double_q=True,
                 n_steps_per_update=1,
                 update_freq=1,
                 target_updates_freq=10000,
                 tau=1,
                 greedy_value=False,
                 # Multistep params
                 n_step_learning=1,
                 # Prioritization params
                 prioritized_sampling=False,
                 alpha=0.6,
                 betta=0.4,
                 priority_eps=1e-3,
                 # Lags params
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first'): 
        
        super().__init__()
        
        # initialize all scheduled args as schedules
        # if they are already not schedules
        self.global_step = 0
        self.local_step = 0
        self.n_updates = 0
        self.n_target_updates = 0 #debug
        self.ckpt_attrs = ['global_step', 'local_step', 'n_updates', 'n_target_updates']
        
        self.gamma = init_schedule(gamma)
        self.tau = init_schedule(tau)
        self.learning_rate = init_schedule(learning_rate)
        self.cql_alpha = init_schedule(cql_alpha)
        self.batch_size = init_schedule(batch_size)
        self.alpha = init_schedule(alpha)
        self.betta = init_schedule(betta)
        self.target_updates_freq = init_schedule(target_updates_freq, discrete=True)
        self.update_freq = init_schedule(update_freq, discrete=True)
        self.n_steps_per_update = init_schedule(n_steps_per_update, discrete=True)
        self.n_step_learning = init_schedule(n_step_learning, discrete=True)
        
        # constant params       
        self.double_q = double_q
        self.greedy_value = greedy_value
        
        self.grad_norm_clipping = grad_norm_clipping
        
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        self.priority_eps = priority_eps
        self.prioritized_sampling = prioritized_sampling
        
        # nn.Module params  
        self.device = device
        self.td_criterion = td_criterion
        # critic net
        self.critic_net = critic_net
        self.critic_net.to(self.device)
        
        self.optimizer = optim.Adam(self.critic_net.parameters(), lr=1, eps=1e-6)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lambda t: self.learning_rate.value(t))
        
        # target net
        self.target_net = deepcopy(critic_net)
        self.target_net.to(self.device)
        
        # detect noisy layers if any
        self.is_noisy = False
        for m in self.critic_net.modules():
            if isinstance(m, NoisyLinear):
                self.is_noisy = True
                break
                
        # loging:
        self.last_logs = {}
        
    def set_device(self, device):
        self.device = device
        self.critic_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        
    def schedules_step(self):
        self.global_step += 1
        if hasattr(self, 'scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
        
    def forward(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        return self.critic_net(obs)
    
    def forward_np(self, obs: np.ndarray) -> np.ndarray:
        obs = from_numpy(self.device, obs)
        return self.forward(obs).cpu().detach().numpy()
            
    def update_target_network(self):
        
        tau = self.tau.value(self.global_step)
        self.n_target_updates += 1
        
        for target_param, param in zip(
            self.target_net.parameters(), self.critic_net.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
    def reset_critic_noise(self):
        for m in self.critic_net.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
                
    def reset_target_noise(self):
        for m in self.target_net.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
        
    def update(self, buffer: ReplayBuffer) -> dict:
        
        logs = {}
        
        # perform training loop
        for _ in range(self.n_steps_per_update.value(self.global_step)):
            
            if (self.learning_rate.value(self.global_step) > 0
                and self.local_step % self.update_freq.value(self.global_step) == 0):

                # sampling self.batch_size transitions:
                batch_size = self.batch_size.value(self.global_step)
                
                # check if prioritization is needed and sample
                sample = buffer.sample(batch_size=batch_size,
                                       p_learner=self if self.prioritized_sampling else None)
                
                # DYNA acceleration if needed
                if hasattr(self, 'acceleration'):
                    sample.accelerate(**self.acceleration_config)
                
                # handling multistep learning and crating next_obs:
                n_steps = self.n_step_learning.value(self.global_step)
                gamma = self.gamma.value(self.global_step)
                rews, gamma_pow, terminals = handle_n_step(data=sample, 
                                                           n=n_steps, 
                                                           gamma=gamma)

                # creating lags if needed in model
                # unpack rollouts for training
                obs, next_obs = handle_lags(data=sample,
                                            fields={'obs': 'lag_concat_obs',
                                                    'next_obs': 'lag_concat_next_obs'},
                                            nlags=self.obs_nlags,
                                            concat_axis=self.obs_concat_axis,
                                            expand_axis=self.obs_expand_axis,
                                            padding=self.obs_padding)

                acs = sample.unpack(['acs'])

                obs = from_numpy(self.device, obs)
                acs = from_numpy(self.device, acs)
                next_obs = from_numpy(self.device, next_obs)
                rews = from_numpy(self.device, rews)
                terminals = from_numpy(self.device, terminals)
                gamma_pow = from_numpy(self.device, gamma_pow)

                # estimate q-values for t
                qa_t_values = self.forward(obs)
                q_t_values = torch.gather(qa_t_values, 1, 
                                              acs.to(torch.long).unsqueeze(1)
                                          ).squeeze(1)

                # estimate q-values for t+1 from target network
                qa_tp1_values = self.target_net(next_obs)

                # estimate action maximized q-values for t+1
                if self.double_q:
                    qa_tp1_values_net = self.critic_net(next_obs)
                    q_tp1_values = torch.gather(qa_tp1_values, 1, 
                                                    qa_tp1_values_net.argmax(1).unsqueeze(1)
                                                ).squeeze(1)
                else:
                    q_tp1_values = torch.gather(qa_tp1_values, 1, 
                                                    qa_tp1_values.argmax(1).unsqueeze(1)
                                                ).squeeze(1)

                # estimate targets
                targets = rews + gamma_pow * q_tp1_values * torch.logical_not(terminals)
                targets = targets.detach()

                # calculate DQN loss
                if self.prioritized_sampling:
                    # prioritized case
                    # compute importance sampling weights MAY BE DO IT INSIDE sample?
                    alpha = self.alpha.value(self.global_step)
                    betta = self.betta.value(self.global_step)
                    
                    p_alpha = sample.unpack(['p_alpha'])
                    p_alpha = from_numpy(self.device, p_alpha)
                    
                    N = sample.parent_buffer.n_transitions
                    p_alpha_total = sample.get_priority_sum(p_learner=self)
                    
                    probs = p_alpha / p_alpha_total
                    
                    is_weights = (N * probs)**(-betta)
                    is_weights = is_weights / is_weights.max()
                    
                    # compute weighted loss
                    elementwise_dqn_loss = self.td_criterion(q_t_values, targets, reduction='none')
                    dqn_loss = torch.mean(elementwise_dqn_loss * is_weights)
                    
                    # update priorities using TD-error
                    new_priorities = to_numpy(elementwise_dqn_loss) + self.priority_eps
                    sample.update_priorities(
                        p_learner=self,
                        p=new_priorities,
                        alpha=alpha
                    )
                    
                else:
                    # uniform case
                    dqn_loss = self.td_criterion(q_t_values, targets)

                # calculate CQL loss regularization
                q_t_logsumexp = qa_t_values.logsumexp(dim=-1) # numerically stabilized
                cql_loss = torch.mean(q_t_logsumexp - q_t_values)

                # calculate total loss
                cql_alpha = self.cql_alpha.value(self.global_step)
                loss = dqn_loss + cql_alpha * cql_loss

                # performing gradient step
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_norm_clipping is not None:
                    torch.nn.utils.clip_grad_value_(self.critic_net.parameters(), self.grad_norm_clipping)
                self.optimizer.step()
                self.n_updates += 1
                
                # reset  noise if needed 
                if self.is_noisy:
                    self.reset_critic_noise()
                    self.reset_target_noise()
                
                # update target network
                if self.n_updates % self.target_updates_freq.value(self.global_step) == 0 and self.n_updates > 0:
                    self.update_target_network()
                
                # logging
                pr = type(self).__name__
                self.last_logs = {f'{pr}_critic_loss': loss.item(),
                                  f'{pr}_critic_dqn_loss': dqn_loss.item(),
                                  f'{pr}_critic_cql_loss': cql_loss.item(),
                                  f'{pr}_gamma': self.gamma.value(self.global_step),
                                  f'{pr}_terminal_share': terminals.mean().item(),
                                  f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                                  f'{pr}_cql_alpha': cql_alpha,
                                  f'{pr}_dqn_batch_size': sample.n_transitions,
                                  f'{pr}_data_q_values': to_numpy(q_t_values).mean(),
                                  f'{pr}_ood_q_values': to_numpy(q_t_logsumexp).mean(),
                                  f'{pr}_global_step': self.global_step,
                                  f'{pr}_local_step': self.local_step,
                                  f'{pr}_n_updates': self.n_updates,
                                  f'{pr}_n_target_updates': self.n_target_updates}
                
                # delete sample
                del sample

            self.local_step += 1
        
        # global step for schedules
        self.schedules_step()
        
        return self.last_logs
    
    def estimate_value(self, paths_or_sample) -> np.ndarray:
        
        obs = handle_lags(data=paths_or_sample,
                          fields={'obs': 'lag_concat_obs'},
                          nlags=self.obs_nlags,
                          concat_axis=self.obs_concat_axis,
                          expand_axis=self.obs_expand_axis,
                          padding=self.obs_padding)
        
        obs = from_numpy(self.device, obs)
        qa_t_values = self.forward(obs)
        
        if self.greedy_value:
            values = torch.gather(qa_t_values, 1, 
                                      qa_t_values.argmax(1).unsqueeze(1)
                                  ).squeeze(1)
        else:
            values = qa_t_values.mean(-1)
        
        return to_numpy(values)
    
    def estimate_qvalue(self, paths_or_sample) -> np.ndarray:
        
        obs = handle_lags(data=paths_or_sample,
                          fields={'obs': 'lag_concat_obs'},
                          nlags=self.obs_nlags,
                          concat_axis=self.obs_concat_axis,
                          expand_axis=self.obs_expand_axis,
                          padding=self.obs_padding)

        acs = paths_or_sample.unpack(['acs'])

        obs = from_numpy(self.device, obs)
        acs = from_numpy(self.device, acs)
        
        qa_t_values = self.forward(obs)
        q_t_values = torch.gather(qa_t_values, 1, 
                                      acs.to(torch.long).unsqueeze(1)
                                  ).squeeze(1)
        
        return to_numpy(q_t_values)
        
    def estimate_advantage(self, paths_or_sample) -> np.ndarray:
        
        q_values = self.estimate_qvalues(paths_or_sample=paths_or_sample)
        values = self.estimate_values(paths_or_sample=paths_or_sample)
        
        advantages = q_values - values
        
        return advantages
    
    
class CategoricalDQN(DQN):
    
    def __init__(self, 
                 critic_net: nn.Module,
                 device,
                 learning_rate,
                 batch_size,
                 gamma=0.99,
                 cql_alpha=0.0,
                 grad_norm_clipping=None,
                 # td_criterion=F.smooth_l1_loss,
                 double_q=True,
                 n_steps_per_update=1,
                 update_freq=1,
                 target_updates_freq=10000,
                 tau=1,
                 greedy_value=False,
                 # Multistep params
                 n_step_learning=1,
                 # Prioritization params
                 prioritized_sampling=False,
                 alpha=0.6,
                 betta=0.4,
                 priority_eps=1e-3,
                 # Lags params
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first'): 
        
        """
        Args:
        
            critic_net: nn.Module
            
            A network taking a batch of obs  
            with a shape (batch_size, *obs_dim)
            and returning a tensor of log probabilities 
            with a shape (batch_size, acs_dim, n_atoms)
            
            ! Note:
            
            critic_net should have these attributes:
            
            v_min - minimal support value of categorical distribution
            v_max - maximal support value of categorical distribution
            n_atoms - number of categorical distribution support elements
            d_z - step of categorical distribution support =
            (v_max-v_min)/(n_atoms-1)
            support: torch.FloatTensor - categorical distribution support
            which should be registered in model's buffer

        
        """
        
        super().__init__(critic_net=critic_net,
                         device=device,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         gamma=gamma,
                         cql_alpha=cql_alpha,
                         grad_norm_clipping=grad_norm_clipping,
                         td_criterion=None, # Not defined for categorical DQN
                         double_q=double_q,
                         n_steps_per_update=n_steps_per_update,
                         update_freq=update_freq,
                         target_updates_freq=target_updates_freq,
                         tau=tau,
                         greedy_value=greedy_value,
                         # Multistep params
                         n_step_learning=n_step_learning,
                         # Prioritization params
                         prioritized_sampling=prioritized_sampling,
                         alpha=alpha,
                         betta=betta,
                         priority_eps=priority_eps,
                         # Lags params
                         obs_nlags=obs_nlags,
                         obs_concat_axis=obs_concat_axis,
                         obs_expand_axis=obs_expand_axis,
                         obs_padding=obs_padding)
        
        del self.td_criterion
        
    @staticmethod
    def get_q_values(log_probs: torch.FloatTensor,
                     module: nn.Module)-> torch.FloatTensor:
        probs = torch.exp(log_probs)
        q_values = torch.sum(probs * module.support, axis=-1)
        return q_values

    def forward(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        log_probs = self.critic_net(obs)
        q_values = self.get_q_values(log_probs=log_probs,
                                     module=self.critic_net)
        return q_values

    def update(self, buffer: ReplayBuffer) -> dict:

        """
        Credits:

        https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/06.categorical_dqn.ipynb

        https://github.com/Kaixhin/Rainbow/blob/master/agent.py

        """

        logs = {}
        
        # perform training loop
        for _ in range(self.n_steps_per_update.value(self.global_step)):
            
            if (self.learning_rate.value(self.global_step) > 0
                and self.local_step % self.update_freq.value(self.global_step) == 0):

                # sampling self.batch_size transitions:
                batch_size = self.batch_size.value(self.global_step)
                
                # check if prioritization is needed and sample
                sample = buffer.sample(batch_size=batch_size,
                                       p_learner=self if self.prioritized_sampling else None)
                
                # DYNA acceleration if needed
                if hasattr(self, 'acceleration'):
                    sample.accelerate(**self.acceleration_config)
                
                # handling multistep learning and crating next_obs:
                n_steps = self.n_step_learning.value(self.global_step)
                gamma = self.gamma.value(self.global_step)
                rews, gamma_pow, terminals = handle_n_step(data=sample, 
                                                           n=n_steps, 
                                                           gamma=gamma)

                # creating lags if needed in model
                # unpack rollouts for training
                obs, next_obs = handle_lags(data=sample,
                                            fields={'obs': 'lag_concat_obs',
                                                    'next_obs': 'lag_concat_next_obs'},
                                            nlags=self.obs_nlags,
                                            concat_axis=self.obs_concat_axis,
                                            expand_axis=self.obs_expand_axis,
                                            padding=self.obs_padding)

                acs = sample.unpack(['acs'])

                obs = from_numpy(self.device, obs)
                acs = from_numpy(self.device, acs)
                next_obs = from_numpy(self.device, next_obs)
                rews = from_numpy(self.device, rews).unsqueeze(1)
                terminals = from_numpy(self.device, terminals).unsqueeze(1)
                gamma_pow = from_numpy(self.device, gamma_pow).unsqueeze(1)
                
                # Implementing categorical DQN update:
                # estimate log distribution for t+1 from target network
                log_next_dist_acs = self.target_net(next_obs)
                
                # estimate action maximized q-values distributions for t+1
                if self.double_q:

                    log_next_dist_acs_net = self.critic_net(next_obs)

                    qa_tp1_values_net = self.get_q_values(
                        log_probs=log_next_dist_acs_net,
                        module=self.critic_net
                    )

                    log_next_dist = log_next_dist_acs[range(sample.n_transitions), 
                                                      qa_tp1_values_net.argmax(1)]
                else:

                    qa_tp1_values = self.get_q_values(
                        log_probs=log_next_dist_acs,
                        module=self.target_net
                    )

                    log_next_dist = log_next_dist_acs[range(sample.n_transitions), 
                                                      qa_tp1_values.argmax(1)]

                next_dist = log_next_dist.exp().detach()
                
                # Compute the projection Tz onto the support z
                t_z = rews + (1 - terminals) * gamma_pow * self.critic_net.support
                t_z = t_z.clamp(min=self.critic_net.v_min, max=self.critic_net.v_max)
                
                b = (t_z - self.critic_net.v_min) / self.critic_net.d_z
                l = b.floor().long()
                u = b.ceil().long()
                
                # Distribute the probability of Tz to the closest neighbours
                m = torch.zeros(next_dist.shape, device=self.device)
                
                offset = torch.linspace(
                    0, (sample.n_transitions - 1) * next_dist.size(1), 
                    sample.n_transitions
                ).long().unsqueeze(1).expand(*next_dist.shape).to(self.device)
                
                # Add to lower neighbour
                m.view(-1).index_add_(
                    0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                )
                
                # Add to upper neighbour
                m.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )
                
                # Estimate critics Q-values distributions for obs & acs
                log_dist_acs = self.critic_net.forward(obs)
                
                # Log distribution
                log_dist = log_dist_acs[range(sample.n_transitions), acs.long()]
                
                # Q-values expectation
                qa_t_values = self.get_q_values(log_probs=log_dist_acs,
                                                module=self.critic_net)
                
                q_t_values = torch.gather(qa_t_values, 1, 
                                              acs.to(torch.long).unsqueeze(1)
                                          ).squeeze(1)
                
                # calculate Categorical DQN loss
                elementwise_dqn_loss = -(m * log_dist).sum(1)
                
                if self.prioritized_sampling:
                    # prioritized case
                    # compute importance sampling weights MAY BE DO IT INSIDE sample?
                    alpha = self.alpha.value(self.global_step)
                    betta = self.betta.value(self.global_step)
                    
                    p_alpha = sample.unpack(['p_alpha'])
                    p_alpha = from_numpy(self.device, p_alpha)
                    
                    N = sample.parent_buffer.n_transitions
                    p_alpha_total = sample.get_priority_sum(p_learner=self)
                    
                    probs = p_alpha / p_alpha_total
                    
                    is_weights = (N * probs)**(-betta)
                    is_weights = is_weights / is_weights.max()
                    
                    # compute weighted loss
                    dqn_loss = torch.mean(elementwise_dqn_loss * is_weights)
                    
                    # update priorities using TD-error
                    new_priorities = to_numpy(elementwise_dqn_loss) + self.priority_eps
                    sample.update_priorities(
                        p_learner=self,
                        p=new_priorities,
                        alpha=alpha
                    )
                    
                else:
                    # uniform case
                    dqn_loss = torch.mean(elementwise_dqn_loss)

                # calculate CQL loss regularization
                q_t_logsumexp = qa_t_values.logsumexp(dim=-1) # numerically stabilized
                cql_loss = torch.mean(q_t_logsumexp - q_t_values)

                # calculate total loss
                cql_alpha = self.cql_alpha.value(self.global_step)
                loss = dqn_loss + cql_alpha * cql_loss

                # performing gradient step
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_norm_clipping is not None:
                    torch.nn.utils.clip_grad_value_(self.critic_net.parameters(), 
                                                    self.grad_norm_clipping)
                self.optimizer.step()
                self.n_updates += 1
                
                # reset noise if needed 
                if self.is_noisy:
                    self.reset_critic_noise()
                    self.reset_target_noise()
                
                # update target network
                if self.n_updates % self.target_updates_freq.value(self.global_step) == 0 and self.n_updates > 0:
                    self.update_target_network()
                
                # logging
                # Calculate Q-value distribution entropy for logging
                log_dist = log_dist.detach()
                dist = log_dist.exp()
                entropy = - dist * log_dist
                entropy = entropy.sum(-1).mean()
                
                pr = type(self).__name__
                self.last_logs = {f'{pr}_critic_loss': loss.item(),
                                  f'{pr}_critic_dqn_loss': dqn_loss.item(),
                                  f'{pr}_critic_cql_loss': cql_loss.item(),
                                  f'{pr}_q_values_dist_entropy': entropy.item(),
                                  f'{pr}_gamma': self.gamma.value(self.global_step),
                                  f'{pr}_terminal_share': terminals.mean().item(),
                                  f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                                  f'{pr}_cql_alpha': cql_alpha,
                                  f'{pr}_dqn_batch_size': sample.n_transitions,
                                  f'{pr}_data_q_values': to_numpy(q_t_values).mean(),
                                  f'{pr}_ood_q_values': to_numpy(q_t_logsumexp).mean(),
                                  f'{pr}_global_step': self.global_step,
                                  f'{pr}_local_step': self.local_step,
                                  f'{pr}_n_updates': self.n_updates,
                                  f'{pr}_n_target_updates': self.n_target_updates}
                
                # delete sample
                del sample

            self.local_step += 1
        
        # global step for schedules
        self.schedules_step()
        
        return self.last_logs

    
class CDQN(nn.Module, Checkpointer, metaclass=abc.ABCMeta):
    
    """
    CDQN - Continuous Deep Q-Network
    """
    
    def __init__(self, 
                 critic_net: nn.Module,
                 device,
                 learning_rate,
                 gamma=0.99,
                 tau=1e-3,
                 weight_decay=1e-4,
                 grad_norm_clipping=None,
                 critic_net2=None,
                 td_criterion=F.mse_loss,
                 # Multistep params
                 n_step_learning=1,
                 # Prioritization params
                 prioritized_sampling=False,
                 alpha=0.6,
                 betta=0.4,
                 priority_eps=1e-3,
                 # Lags params
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first',
                 **kwargs): 
        
        super().__init__(**kwargs)
        
        from relax.rl.actors import DDPG, TD3, SAC
        
        # initialize all scheduled args as schedules
        # if they are already not schedules
        self.global_step = 0
        self.local_step = 0
        self.n_updates = 0
        self.n_target_updates = 0
        self.ckpt_attrs = ['global_step', 'local_step', 'n_updates', 'n_target_updates']
        
        self.tau = init_schedule(tau)
        self.gamma = init_schedule(gamma)
        self.alpha = init_schedule(alpha)
        self.betta = init_schedule(betta)
        self.learning_rate = init_schedule(learning_rate)
        self.n_step_learning = init_schedule(n_step_learning, discrete=True)
        
        # constant params 
        self.grad_norm_clipping = grad_norm_clipping
        
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        self.priority_eps = priority_eps
        self.prioritized_sampling = prioritized_sampling
        
        # nn.Module params  
        self.device = device
        self.td_criterion = td_criterion
        # critic net
        self.critic_net = critic_net
        self.critic_net.to(self.device)
        
        self.optimizer = optim.Adam(self.critic_net.parameters(), 
                                    lr=1, 
                                    eps=1e-6,
                                    weight_decay=weight_decay) 
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lambda t: self.learning_rate.value(t))
        
        # target net
        self.target_net = deepcopy(critic_net)
        self.target_net.to(self.device)
        
        # loging:
        self.last_logs = {}
        
        # twin critic net for TD3 & SAC setting if needed:
        self.critic_net2 = critic_net2
        if isinstance(self.critic_net2, nn.Module):
            
            self.critic_net2.to(self.device)
            self.optimizer2 = optim.Adam(self.critic_net2.parameters(), 
                                         lr=1, 
                                         eps=1e-6,
                                         weight_decay=weight_decay) 
            self.scheduler2 = optim.lr_scheduler.LambdaLR(self.optimizer2,
                                                          lambda t: self.learning_rate.value(t))
            
            # target net
            self.target_net2 = deepcopy(critic_net2)
            self.target_net2.to(self.device)
        
        else:
            self.critic_net2 = None
            self.target_net2 = None
            
        self.valid_actors = {'DDPG': DDPG,
                               'TD3': TD3,
                               'SAC': SAC}
        
    
    def set_device(self, device):
        self.device = device
        self.critic_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        if isinstance(self.critic_net2, nn.Module):
            self.critic_net2.to(self.device)
            self.target_net2.to(self.device)
            self.optimizer2.load_state_dict(self.optimizer2.state_dict())
        
    def schedules_step(self):
        self.global_step += 1
        if hasattr(self, 'scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
        if hasattr(self, 'scheduler2'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler2.step()
        
    def forward(self, obs: torch.FloatTensor,
                acs: torch.FloatTensor, target=False) -> torch.FloatTensor:
        if target:
            return self.target_net(obs=obs, acs=acs)
        else:
            return self.critic_net(obs=obs, acs=acs)
            
    def forward_np(self, obs: np.ndarray,
                   acs: np.ndarray, target=False) -> np.ndarray:
        obs = from_numpy(self.device, obs)
        acs = from_numpy(self.device, acs)
        qvals = self.forward(obs=obs, acs=acs, target=target)
        return to_numpy(qvals)
    
    def update_target_network(self):
        
        tau = self.tau.value(self.global_step)
        self.n_target_updates += 1
        
        for target_param, param in zip(
            self.target_net.parameters(), self.critic_net.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        if isinstance(self.critic_net2, nn.Module):
            for target_param2, param2 in zip(
                self.target_net2.parameters(), self.critic_net2.parameters()
            ):
                target_param2.data.copy_(tau * param2.data + (1 - tau) * target_param2.data)
            
    def _ddpg_update(self, 
                     obs: np.ndarray,
                     next_obs: np.ndarray,
                     acs: np.ndarray, 
                     rews: np.ndarray,
                     gamma_pow: np.ndarray,
                     terminals: np.ndarray,
                     is_weights: np.ndarray,
                     actor_next_obs: np.ndarray,
                     actor,
                     sample: BufferSample):
        
        # check if the actor of right type
        if not isinstance(actor, self.valid_actors['DDPG']):
            raise ValueError(
                f'Invalid actor type {type(actor).__name__} for {type(self).__name__}, it should be DDPG.'
            )
        
        # Transfer the data to required device
        obs = from_numpy(self.device, obs)
        acs = from_numpy(self.device, acs)
        next_obs = from_numpy(self.device, next_obs)
        rews = from_numpy(self.device, rews)
        gamma_pow = from_numpy(self.device, gamma_pow)
        terminals = from_numpy(self.device, terminals)
        actor_next_obs = from_numpy(actor.device, actor_next_obs)
        
        # constructing the target value for Bellman error:     
        next_acs_target = actor.forward(
            obs=actor_next_obs,
            target=True
        )
        
        q_tp1_target = self.forward(obs=next_obs, 
                                    acs=next_acs_target.to(self.device), 
                                    target=True).squeeze()
        
        targets = rews + gamma_pow * q_tp1_target * torch.logical_not(terminals)
        targets = targets.detach()
        
        # calculate CDQN loss
        q_t_values = self.forward(obs=obs, acs=acs).squeeze()
        
        if self.prioritized_sampling:         
            # Compute weighted loss
            elementwise_cdqn_loss = self.td_criterion(q_t_values, targets, reduction='none')
            is_weights = from_numpy(self.device, is_weights)
            cdqn_loss = torch.mean(elementwise_cdqn_loss*is_weights)
            
            # update priorities using TD-error
            new_priorities = to_numpy(elementwise_cdqn_loss) + self.priority_eps
            sample.update_priorities(
                p_learner=self,
                p=new_priorities,
                alpha=self.alpha.value(self.global_step)
            )
                    
        else:
            # Uniform case
            cdqn_loss = self.td_criterion(q_t_values, targets)
        
        # Performing gradient step
        self.optimizer.zero_grad()
        cdqn_loss.backward()
        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_value_(self.critic_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.n_updates += 1
        self.local_step += 1
        
        # logging
        pr = type(self).__name__
        self.last_logs = {f'{pr}_critic_loss': cdqn_loss.item(),
                          f'{pr}_tau': self.tau.value(self.global_step),
                          f'{pr}_gamma': self.gamma.value(self.global_step),
                          f'{pr}_terminal_share': terminals.mean().item(),
                          f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                          f'{pr}_data_q_values': to_numpy(q_t_values).mean(),
                          f'{pr}_global_step': self.global_step,
                          f'{pr}_local_step': self.local_step,
                          f'{pr}_n_updates': self.n_updates,
                          f'{pr}_n_target_updates': self.n_target_updates}
    
    def _td3_update(self, 
                    obs: np.ndarray,
                    next_obs: np.ndarray,
                    acs: np.ndarray, 
                    rews: np.ndarray,
                    gamma_pow: np.ndarray,
                    terminals: np.ndarray,
                    is_weights: np.ndarray,
                    actor_next_obs: np.ndarray,
                    actor,
                    sample: BufferSample):
        
        # check if the actor of right type
        if not isinstance(actor, self.valid_actors['TD3']):
            raise ValueError(
                f'Invalid actor type {type(actor).__name__} for {type(self).__name__}, it should be TD3.'
            )
            
        # check if twin critic is provided
        if not isinstance(self.critic_net2, nn.Module):
            raise AttributeError(
                f'Provide torch.nn.Module twin critic for {type(actor).__name__} setting'
            )

        # Transfer the data to required device
        obs = from_numpy(self.device, obs)
        acs = from_numpy(self.device, acs)
        next_obs = from_numpy(self.device, next_obs)
        rews = from_numpy(self.device, rews)
        gamma_pow = from_numpy(self.device, gamma_pow)
        terminals = from_numpy(self.device, terminals)
        actor_next_obs = from_numpy(actor.device, actor_next_obs)
        
        # constructing the target value for Bellman error:
        next_acs_target = actor.forward_target_smoothed(
            obs=actor_next_obs
        ).to(self.device)
        
        # Target Critic #1
        q_tp1_target1 = self.forward(obs=next_obs, 
                                     acs=next_acs_target, 
                                     target=True).squeeze()
        
        # Target Critic #2
        q_tp1_target2 = self.target_net2(obs=next_obs,
                                         acs=next_acs_target).squeeze()
        
        q_tp1_target_min = torch.min(q_tp1_target1, q_tp1_target2)
        
        targets = rews + gamma_pow * q_tp1_target_min * torch.logical_not(terminals)
        targets = targets.detach()
        
        # calculate CDQN losses
        q_t_values1 = self.forward(obs=obs, acs=acs).squeeze()
        q_t_values2 = self.critic_net2(obs=obs, acs=acs).squeeze()
        
        if self.prioritized_sampling:         
            # Compute weighted loss
            elementwise_cdqn_loss1 = self.td_criterion(q_t_values1, targets, reduction='none')
            elementwise_cdqn_loss2 = self.td_criterion(q_t_values2, targets, reduction='none')
            is_weights = from_numpy(self.device, is_weights)
            cdqn_loss1 = torch.mean(elementwise_cdqn_loss1*is_weights)
            cdqn_loss2 = torch.mean(elementwise_cdqn_loss2*is_weights)
            
            # update priorities using TD-error
            new_priorities = np.mean( # May be use other agg funcs?
                [to_numpy(elementwise_cdqn_loss1), to_numpy(elementwise_cdqn_loss2)],
                axis=0
            ) + self.priority_eps
            sample.update_priorities(
                p_learner=self,
                p=new_priorities,
                alpha=self.alpha.value(self.global_step)
            )
                    
        else:
            # Uniform case
            cdqn_loss1 = self.td_criterion(q_t_values1, targets)
            cdqn_loss2 = self.td_criterion(q_t_values2, targets)
        
        # Performing gradient step for both critics
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        cdqn_loss1.backward()
        cdqn_loss2.backward()
        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_value_(self.critic_net.parameters(), self.grad_norm_clipping)
            torch.nn.utils.clip_grad_value_(self.critic_net2.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.optimizer2.step()
        self.n_updates += 1
        self.local_step += 1
        
        # logging
        pr = type(self).__name__
        self.last_logs = {f'{pr}_critic_loss': cdqn_loss1.item(),
                          f'{pr}_critic_loss2': cdqn_loss2.item(),
                          f'{pr}_tau': self.tau.value(self.global_step),
                          f'{pr}_gamma': self.gamma.value(self.global_step),
                          f'{pr}_terminal_share': terminals.mean().item(),
                          f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                          f'{pr}_data_q_values': to_numpy(q_t_values1).mean(),
                          f'{pr}_data_q_values2': to_numpy(q_t_values2).mean(),
                          f'{pr}_global_step': self.global_step,
                          f'{pr}_local_step': self.local_step,
                          f'{pr}_n_updates': self.n_updates,
                          f'{pr}_n_target_updates': self.n_target_updates}
        
    def _sac_update(self, 
                    obs: np.ndarray,
                    next_obs: np.ndarray,
                    acs: np.ndarray, 
                    rews: np.ndarray,
                    gamma_pow: np.ndarray,
                    terminals: np.ndarray,
                    is_weights: np.ndarray,
                    actor_next_obs: np.ndarray,
                    actor,
                    sample: BufferSample):
        
        # check if the actor of right type
        if not isinstance(actor, self.valid_actors['SAC']):
            raise ValueError(
                f'Invalid actor type {type(actor).__name__} for {type(self).__name__}, it should be SAC.'
            )
            
        # check if twin critic is provided
        if not isinstance(self.critic_net2, nn.Module):
            raise AttributeError(
                f'Provide torch.nn.Module twin critic for {type(actor).__name__} setting'
            )

        # Transfer the data to required device
        obs = from_numpy(self.device, obs)
        acs = from_numpy(self.device, acs)
        next_obs = from_numpy(self.device, next_obs)
        rews = from_numpy(self.device, rews)
        gamma_pow = from_numpy(self.device, gamma_pow)
        terminals = from_numpy(self.device, terminals)
        actor_next_obs = from_numpy(actor.device, actor_next_obs)
        
        # constructing the target value for Bellman error:
        alpha = actor.alpha
        next_acs_dist = actor.forward(
            obs=actor_next_obs
        )
        next_acs_sample = next_acs_dist.sample().to(self.device)
        
        # Target Critic #1
        q_tp1_target1 = self.forward(obs=next_obs, 
                                     acs=next_acs_sample, 
                                     target=True).squeeze()
        
        # Target Critic #2
        q_tp1_target2 = self.target_net2(obs=next_obs,
                                         acs=next_acs_sample).squeeze()
        
        q_tp1_target_min = torch.min(q_tp1_target1, q_tp1_target2)
        
        # Adding the log_prob
        log_prob = next_acs_dist.log_prob(
            next_acs_sample.to(actor.device)
        ).detach().to(self.device) # Remove detach here?
        
        q_tp1_ent = q_tp1_target_min - alpha * log_prob
        
        # Formulating the targets
        targets = rews + gamma_pow * q_tp1_ent * torch.logical_not(terminals)
        targets = targets.detach()
        
        # calculate CDQN losses
        q_t_values1 = self.forward(obs=obs, acs=acs).squeeze()
        q_t_values2 = self.critic_net2(obs=obs, acs=acs).squeeze()
        
        if self.prioritized_sampling:         
            # Compute weighted loss
            elementwise_cdqn_loss1 = self.td_criterion(q_t_values1, targets, reduction='none')
            elementwise_cdqn_loss2 = self.td_criterion(q_t_values2, targets, reduction='none')
            is_weights = from_numpy(self.device, is_weights)
            cdqn_loss1 = torch.mean(elementwise_cdqn_loss1*is_weights)
            cdqn_loss2 = torch.mean(elementwise_cdqn_loss2*is_weights)
            
            # update priorities using TD-error
            new_priorities = np.mean( # May be use other agg funcs?
                [to_numpy(elementwise_cdqn_loss1), to_numpy(elementwise_cdqn_loss2)],
                axis=0
            ) + self.priority_eps
            sample.update_priorities(
                p_learner=self,
                p=new_priorities,
                alpha=self.alpha.value(self.global_step)
            )
                    
        else:
            # Uniform case
            cdqn_loss1 = self.td_criterion(q_t_values1, targets)
            cdqn_loss2 = self.td_criterion(q_t_values2, targets)
        
        # Performing gradient step for both critics
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        cdqn_loss1.backward()
        cdqn_loss2.backward()
        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_value_(self.critic_net.parameters(), self.grad_norm_clipping)
            torch.nn.utils.clip_grad_value_(self.critic_net2.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.optimizer2.step()
        self.n_updates += 1
        self.local_step += 1
        
        # logging
        pr = type(self).__name__
        self.last_logs = {f'{pr}_critic_loss': cdqn_loss1.item(),
                          f'{pr}_critic_loss2': cdqn_loss2.item(),
                          f'{pr}_tau': self.tau.value(self.global_step),
                          f'{pr}_gamma': self.gamma.value(self.global_step),
                          f'{pr}_terminal_share': terminals.mean().item(),
                          f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                          f'{pr}_data_q_values': to_numpy(q_t_values1).mean(),
                          f'{pr}_data_q_values2': to_numpy(q_t_values2).mean(),
                          f'{pr}_global_step': self.global_step,
                          f'{pr}_local_step': self.local_step,
                          f'{pr}_n_updates': self.n_updates,
                          f'{pr}_n_target_updates': self.n_target_updates}
        
    def update(self, buffer=None):
        
        """
        This method does not perform an update of
        critics networks as DDPG/TD3/SAC critics
        should be updated on the same batch of data 
        simultaneously
        
        This method updates critic's scheduler and
        created to follow general training loop interface
        """
        
        self.schedules_step()
        return self.last_logs
    
    def estimate_value(self, paths_or_sample, actor) -> np.ndarray:
        
        assert isinstance(actor, list(self.valid_actors.values()))
        # What in case of SAC?
        
        obs = handle_lags(data=paths_or_sample,
                          fields={'obs': 'lag_concat_obs'},
                          nlags=self.obs_nlags,
                          concat_axis=self.obs_concat_axis,
                          expand_axis=self.obs_expand_axis,
                          padding=self.obs_padding)
        
        obs = from_numpy(self.device, obs)
        
        best_acs = actor.forward(
            obs=obs.to(actor.device)
        ).to(self.device)
        
        if isinstance(actor, self.valid_actors['SAC']):
            if hasattr(best_acs, 'deterministic_acs'):
                best_acs = best_acs.deterministic_acs
            else:
                best_acs = best_acs.sample()
            
        values = self.forward(obs=obs, acs=best_acs).squeeze()
        
        return to_numpy(values)
    
    def estimate_qvalue(self, paths_or_sample) -> np.ndarray:
        
        obs = handle_lags(data=paths_or_sample,
                          fields={'obs': 'lag_concat_obs'},
                          nlags=self.obs_nlags,
                          concat_axis=self.obs_concat_axis,
                          expand_axis=self.obs_expand_axis,
                          padding=self.obs_padding)
        
        acs = paths_or_sample.unpack(['acs'])
        
        qvalues = self.forward_np(obs=obs, acs=acs).squeeze()
        
        return qvalues
                
    def estimate_advantage(self, paths_or_sample, actor) -> np.ndarray:
        
        values = self.estimate_value(paths_or_sample=paths_or_sample,
                                     actor=actor)
        
        qvalues = self.estimate_qvalue(paths_or_sample=paths_or_sample)
        
        advantages = qvalues - values
        
        return advantages
    
