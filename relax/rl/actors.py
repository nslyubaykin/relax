import abc
import torch
import warnings
import numpy as np

from copy import deepcopy
from itertools import accumulate

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from relax.data.sampling import PathList
from relax.data.replay_buffer import ReplayBuffer
from relax.data.utils import normalize, handle_lags, handle_n_step
from relax.schedules import init_schedule
from relax.data.acceleration import DynaAccelerator
from relax.torch.utils import *


class BaseActor(Checkpointer, metaclass=abc.ABCMeta):
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError
        

class Random(BaseActor):
    
    def __init__(self, env):
        self.env = env
        self.train_sampling = False
        
        self.obs_nlags = 0
        self.obs_concat_axis = -1
        self.obs_expand_axis = None
        self.obs_padding = 'zeros'
    
    def get_action(self, obs):
        assert len(obs.shape) >= 2
        out_acs = []
        for _ in range(obs.shape[0]):
            out_acs.append(self.env.action_space.sample())
        return np.array(out_acs)
    

class RandomUniform(BaseActor):
    
    def __init__(self, acs_dim,
                 min_acs, max_acs):
        
        self.acs_dim =acs_dim
        self.min_acs = min_acs
        self.max_acs = max_acs
        
        self.train_sampling = False
        
        self.obs_nlags = 0
        self.obs_concat_axis = -1
        self.obs_expand_axis = None
        self.obs_padding = 'zeros'
    
    def get_action(self, obs):
        assert len(obs.shape) >= 2
        out_acs = np.random.uniform(low=self.min_acs, 
                                    high=self.max_acs, 
                                    size=(obs.shape[0], self.acs_dim))
        return out_acs

    
class VPG(BaseActor, nn.Module, metaclass=abc.ABCMeta):
    
    def __init__(self, 
                 policy_net: nn.Module, # A network that controls policy
                 device,
                 learning_rate,
                 n_steps_per_update=1,
                 gamma=0.99,
                 standardize_advantages=True, 
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.train_sampling = False
        
        # initialize counters
        self.global_step = 0
        self.n_updates = 0
        self.ckpt_attrs = ['global_step', 'n_updates']
        
        # initialize schedules
        self.gamma = init_schedule(gamma)
        self.learning_rate = init_schedule(learning_rate)
        self.n_steps_per_update = init_schedule(n_steps_per_update, discrete=True) # changed the name!
        
        # initialize torch objects
        self.device = device
        self.policy_net = policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1, eps=1e-6)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lambda t: self.learning_rate.value(t))
        self.policy_net.to(self.device)
        
        # initialize constant params
        self.standardize_advantages = standardize_advantages
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        # other params
        self.critic = None
        self.last_logs = {}
        
    def set_critic(self, critic):
        self.critic = critic
        
    def set_device(self, device):
        self.device = device
        self.policy_net.to(self.device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = from_numpy(self.device, obs)
        acs = self.policy_net(obs)
        return acs.sample().detach().cpu().numpy()
    
    def forward(self, obs: torch.FloatTensor) -> torch.distributions.distribution.Distribution:
        return self.policy_net(obs)
    
    def schedules_step(self):
        self.global_step += 1
        if hasattr(self, 'scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
    
    def update(self, paths: PathList) -> dict:
        
        # creating lags if needed in model
        # unpack rollouts for training
        obs = handle_lags(data=paths,
                          fields={'obs': 'lag_concat_obs'},
                          nlags=self.obs_nlags,
                          concat_axis=self.obs_concat_axis,
                          expand_axis=self.obs_expand_axis,
                          padding=self.obs_padding)
            
        acs = paths.unpack(['acs'])
        
        # check if any critic provided calculated advantages:
        if self.critic is not None:
            # if true - use these pre calculated advantages
            advantages = self.critic.estimate_advantage(paths)
        else:
            # if false - just use rewards to go as advantages  
            if 'rews_to_go' not in paths.rollouts[0].data.keys():
                paths.add_disc_cumsum('rews_to_go', 'rews',
                                      self.gamma.value(self.global_step))
            advantages = paths.unpack(['rews_to_go'])
        
        # standardize advantages for stability    
        if self.standardize_advantages:
            advantages = normalize(advantages, advantages.mean(), advantages.std())
        
        # convert data to torch.FloatTensor for learning
        obs = from_numpy(self.device, obs)
        acs = from_numpy(self.device, acs)
        advantages = from_numpy(self.device, advantages)
        
        # saving initial distribution for importance sampling:
        # calculate old policy's log prob
        fixed_old_policy = detach_dist(self.forward(obs))
        fixed_log_prob = fixed_old_policy.log_prob(acs).clone()
        init_probs = torch.exp(fixed_log_prob)
        
        # starting policy training loop
        for _ in range(self.n_steps_per_update.value(self.global_step)):
            
            # forward pass to get iteration's policy
            policy = self.forward(obs)
            
            # calculating weights for importtance sampling
            iter_probs = torch.exp(policy.log_prob(acs)).detach() # what if not detach? 
            imp_sampl_weights = iter_probs / (init_probs + 1e-8)
            
            # policy gradient loss function
            loss = - imp_sampl_weights * policy.log_prob(acs) * advantages
            loss = torch.mean(loss)
            
            # performing gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                       
            self.n_updates += 1
            
        # calculate updated policy for logging
        new_policy = detach_dist(self.forward(obs))
        
        pr = type(self).__name__
        self.last_logs = {f'{pr}_actor_loss': loss.item(),
                          f'{pr}_global_step': self.global_step,
                          f'{pr}_n_updates': self.n_updates,
                          f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                          f'{pr}_gamma': self.gamma.value(self.global_step),
                          f'{pr}_n_steps_per_update': self.n_steps_per_update.value(self.global_step),
                          f'{pr}_avg_step_kl_div': torch.mean(kl_divergence(fixed_old_policy, new_policy)).item(),
                          f'{pr}_avg_policy_entropy': torch.mean(new_policy.entropy()).item()}
                       
        self.schedules_step()
        
        return self.last_logs

    
class TRPO(BaseActor, nn.Module, metaclass=abc.ABCMeta):
    
    def __init__(self, 
                 policy_net: nn.Module,
                 device,
                 gamma=0.99,
                 standardize_advantages=True,
                 eps=0.01,
                 damping=0.01,
                 ent_coef=0.0,
                 cg_iters=10,
                 cg_fvp_subsample=1.0,
                 cg_tolerance=1e-8,
                 ls_accept_ratio=0.1,
                 ls_max_backtracks=10,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.train_sampling = False
                       
        # initialize counters
        self.global_step = 0
        self.n_updates = 0
        self.ckpt_attrs = ['global_step', 'n_updates']
        
        # initialize schedules
        self.gamma = init_schedule(gamma)
        self.eps = init_schedule(eps)
        self.damping = init_schedule(damping)
        self.ent_coef = init_schedule(ent_coef)
                               
        # initialize torch objects   
        self.device = device
        self.policy_net = policy_net
        self.policy_net.to(self.device)
        
        # initialize constant params
        self.standardize_advantages = standardize_advantages
        
        self.cg_fvp_subsample = min(1.0, cg_fvp_subsample)         
        self.cg_tolerance = cg_tolerance    
        self.cg_iters = cg_iters
        
        self.ls_accept_ratio = ls_accept_ratio
        self.ls_max_backtracks = ls_max_backtracks
        
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        # other params
        self.critic = None
        self.last_logs = {}
    
    def mean_kl_divergence(self, model: nn.Module, obs: torch.FloatTensor):
        p = model(obs)
        p = detach_dist(p)
        q = self.policy_net(obs)
        return torch.mean(kl_divergence(p, q))
    
    def fisher_vector_product(self, vector: torch.FloatTensor, 
                              obs: torch.FloatTensor,
                              subsample=None) -> torch.FloatTensor:
        
        if subsample is not None and subsample < 1.0:              
            indices = torch.randperm(len(obs))[:round(len(obs) * subsample)]
            obs = obs[indices]

        self.policy_net.zero_grad()
        mean_kl_div = self.mean_kl_divergence(model=self.policy_net, obs=obs)
        kl_grad = torch.autograd.grad(mean_kl_div, 
                                      self.policy_net.parameters(), 
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.reshape(-1) for grad in kl_grad])
        grad_vector_product = torch.sum(kl_grad_vector * vector)
        grad_grad = torch.autograd.grad(grad_vector_product, 
                                        self.policy_net.parameters())
        fisher_vector_product = torch.cat([grad.reshape(-1) for grad in grad_grad]).data
        
        return fisher_vector_product + (self.damping.value(self.global_step) * vector)
    
    def forward(self, obs: torch.FloatTensor) -> torch.distributions.distribution.Distribution:
        return self.policy_net(obs)
        
    def set_critic(self, critic):
        self.critic = critic
                       
    def set_device(self, device):
        self.device = device
        self.policy_net.to(self.device)
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = from_numpy(self.device, obs)
        acs = self.policy_net(obs)
        return acs.sample().detach().cpu().numpy()
                       
    def schedules_step(self):
        self.global_step += 1
        if hasattr(self, 'scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
        
    def surrogate_loss(self, obs: torch.FloatTensor, 
                       acs: torch.FloatTensor, 
                       advantages: torch.FloatTensor,
                       fixed_log_prob: torch.FloatTensor,
                       volatile=False):
        if volatile:
            with torch.no_grad():
                dist = self.forward(obs)
        else:
            dist = self.forward(obs)
        
        log_prob = dist.log_prob(acs)

        loss = - advantages * torch.exp(log_prob - fixed_log_prob)
        loss = torch.mean(loss) - self.ent_coef.value(self.global_step) * torch.mean(dist.entropy())
        
        return loss
    
    def conjugate_gradient(self, b: torch.FloatTensor, 
                           obs: torch.FloatTensor):
        cg_steps = 0
        x = torch.zeros_like(b)
        r, d = b.clone().data, b.clone().data
        rtr = torch.dot(r, r)
        for _ in range(self.cg_iters):
            Ad = self.fisher_vector_product(vector=Variable(d),
                                            obs=obs,
                                            subsample=self.cg_fvp_subsample)
            alpha = rtr / torch.dot(d, Ad)
            x += alpha * d
            r -= alpha * Ad
            rtr_new = torch.dot(r, r)
            betta = rtr_new / rtr
            d = r + betta * d
            rtr = rtr_new
            cg_steps += 1
            if rtr < self.cg_tolerance:
                break
        return cg_steps, x
    
    def linesearch(self, obs: torch.FloatTensor, 
                   acs: torch.FloatTensor, 
                   advantages: torch.FloatTensor,
                   fixed_log_prob: torch.FloatTensor,
                   theta, fullstep, expected_improve_rate):
        
        ls_steps = 0
        fval = self.surrogate_loss(obs=obs, acs=acs, advantages=advantages,
                                   fixed_log_prob=fixed_log_prob,
                                   volatile=True)
        
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(self.ls_max_backtracks)):
            # calculate candidate parameters
            theta_new = theta + stepfrac * fullstep
            # assign them to a model
            vector_to_parameters(vec=theta_new, parameters=self.policy_net.parameters())
            fval_new = self.surrogate_loss(obs=obs, acs=acs, advantages=advantages,
                                           fixed_log_prob=fixed_log_prob,
                                           volatile=True)
            # calculate the improve
            actual_improve = fval - fval_new
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            # check if the improve is sufficient
            ls_steps += 1
            if ratio.item() > self.ls_accept_ratio and actual_improve.item() > 0:
                return True, ls_steps, theta_new
        return False, ls_steps, theta
            
    def update(self, paths: PathList) -> dict:
        
        # creating lags if needed in model
        # unpack rollouts for training
        obs = handle_lags(data=paths,
                          fields={'obs': 'lag_concat_obs'},
                          nlags=self.obs_nlags,
                          concat_axis=self.obs_concat_axis,
                          expand_axis=self.obs_expand_axis,
                          padding=self.obs_padding)
            
        acs = paths.unpack(['acs'])
        
        # check if any critic provided calculated advantages:
        if self.critic is not None:
            # if true - use these pre calculated advantages
            advantages = self.critic.estimate_advantage(paths)
        else:
            # if false - just use rewards to go as advantages  
            if 'rews_to_go' not in paths.rollouts[0].data.keys():
                paths.add_disc_cumsum('rews_to_go', 'rews',
                                      self.gamma.value(self.global_step))
            advantages = paths.unpack(['rews_to_go'])
        
        # standardize advantages for stability    
        if self.standardize_advantages:
            advantages = normalize(advantages, advantages.mean(), advantages.std())
        
        # convert data to torch.FloatTensor for learning
        obs = from_numpy(self.device, obs)
        acs = from_numpy(self.device, acs)
        advantages = from_numpy(self.device, advantages)
        
        # performing TRPO update
        # calculate old policy's log prob
        fixed_old_policy = detach_dist(self.forward(obs))
        fixed_log_prob = fixed_old_policy.log_prob(acs).clone()
        # estimate surrogate loss
        loss = self.surrogate_loss(obs=obs, acs=acs, advantages=advantages,
                                   fixed_log_prob=fixed_log_prob)
        
        # estimate policy gradient
        self.policy_net.zero_grad()
        loss.backward(retain_graph=True)
        policy_gradient = parameters_to_vector(
            [p.grad for p in self.policy_net.parameters()]
        ).squeeze(0)
        
        # compute search direction with CG
        cg_steps, search_dir = self.conjugate_gradient(-policy_gradient, obs=obs)
        
        # compute maximal step length betta
        stAs = torch.dot(search_dir, 
                         self.fisher_vector_product(vector=search_dir, obs=obs))
        betta = torch.sqrt((2 * self.eps.value(self.global_step)) / stAs)
        
        # compute betta * search_dir
        fullstep = betta * search_dir
        neggdotstepdir = torch.dot(-policy_gradient, search_dir)
        
        # performing gradient update
        self.policy_net.zero_grad()
        theta = parameters_to_vector(self.policy_net.parameters()).clone().data
        success, ls_steps, theta_new = self.linesearch(obs=obs, acs=acs, advantages=advantages,
                                                       fixed_log_prob=fixed_log_prob,
                                                       theta=theta, fullstep=fullstep, 
                                                       expected_improve_rate=neggdotstepdir * betta)
        # assign new parameters to policy
        vector_to_parameters(vec=theta_new, parameters=self.policy_net.parameters())
        self.n_updates += 1
        
        # calculate updated policy for logging
        new_policy = detach_dist(self.forward(obs))
        
        pr = type(self).__name__
        self.last_logs = {f'{pr}_actor_loss': loss.item(),
                          f'{pr}_betta': betta.item(),
                          f'{pr}_gamma': self.gamma.value(self.global_step),
                          f'{pr}_eps': self.eps.value(self.global_step),
                          f'{pr}_damping': self.damping.value(self.global_step),
                          f'{pr}_ent_coef': self.ent_coef.value(self.global_step),
                          f'{pr}_avg_step_kl_div': torch.mean(kl_divergence(fixed_old_policy, new_policy)).item(),
                          f'{pr}_avg_policy_entropy': torch.mean(new_policy.entropy()).item(),
                          f'{pr}_cg_steps': cg_steps,
                          f'{pr}_ls_steps': ls_steps,
                          f'{pr}_step_success': success}
                       
        self.schedules_step()
        
        return self.last_logs  
    
    
class PPO(BaseActor, nn.Module, metaclass=abc.ABCMeta):
    
    def __init__(self, 
                 policy_net: nn.Module,
                 device,
                 learning_rate,
                 n_epochs_per_update,
                 batch_size,
                 target_kl=0.01,
                 dataloader_n_workers=0,
                 eps=0.2,
                 grad_norm_clipping=0.5,
                 ent_coef=0.0,
                 gamma=0.99,
                 standardize_advantages=True,
                 weight_decay=0.0,
                 # RND params
                 weight_i=0,
                 weight_e=1,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.train_sampling = False
                       
        # initialize counters
        self.global_step = 0
        self.n_updates = 0
        self.ckpt_attrs = ['global_step', 'n_updates']
        
        # initialize schedules
        self.learning_rate = init_schedule(learning_rate)
        self.gamma = init_schedule(gamma)
        self.target_kl = init_schedule(target_kl)
        self.ent_coef = init_schedule(ent_coef)
        self.eps = init_schedule(np.abs(eps))
        self.weight_i = init_schedule(weight_i)
        self.weight_e = init_schedule(weight_e)
        
        self.n_epochs_per_update = init_schedule(n_epochs_per_update, discrete=True)
        self.batch_size = init_schedule(batch_size, discrete=True)
                               
        # initialize torch objects   
        self.device = device
        self.policy_net = policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=1, 
                                    eps=1e-6,
                                    weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lambda t: self.learning_rate.value(t))
        self.policy_net.to(self.device)
               
        # initialize constant params
        self.standardize_advantages = standardize_advantages
        self.dataloader_n_workers = dataloader_n_workers
        self.grad_norm_clipping = grad_norm_clipping
        
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        # other params
        self.critic = None
        self.exp_critic = None
        self.last_logs = {}
    
    def forward(self, obs: torch.FloatTensor) -> torch.distributions.distribution.Distribution:
        return self.policy_net(obs)
        
    def set_critic(self, critic):
        self.critic = critic
        
    def set_exploration_critic(self, critic):
        self.exp_critic = critic
        if self.exp_critic is not None:
            self.exp_critic.exp_pr = 'Exploration'
                       
    def set_device(self, device):
        self.device = device
        self.policy_net.to(self.device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = from_numpy(self.device, obs)
        acs = self.policy_net(obs)
        return acs.sample().detach().cpu().numpy()
                       
    def schedules_step(self):
        self.global_step += 1
        if hasattr(self, 'scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
        
    def surrogate_loss(self, obs: torch.FloatTensor, 
                       acs: torch.FloatTensor, 
                       advantages: torch.FloatTensor,
                       fixed_log_prob: torch.FloatTensor):
        
        dist = self.forward(obs)
        log_prob = dist.log_prob(acs)
        prob_ratio = torch.exp(log_prob - fixed_log_prob)
        
        eps = self.eps.value(self.global_step)
        
        loss = - torch.min(
            prob_ratio * advantages,
            torch.clamp(prob_ratio, min=1-eps, max=1+eps) * advantages
        )
        
        mean_entropy = torch.mean(dist.entropy())

        loss = torch.mean(loss) - self.ent_coef.value(self.global_step) * mean_entropy
        
        return loss, mean_entropy.detach().item()
            
    def update(self, paths: PathList) -> dict:
        
        # creating lags if needed in model
        # unpack rollouts for training
        obs = handle_lags(data=paths,
                          fields={'obs': 'lag_concat_obs'},
                          nlags=self.obs_nlags,
                          concat_axis=self.obs_concat_axis,
                          expand_axis=self.obs_expand_axis,
                          padding=self.obs_padding)
            
        acs = paths.unpack(['acs'])
        
        # check if any critic provided calculated advantages:
        if self.critic is not None:
            # if true - use these pre calculated advantages
            advantages = self.critic.estimate_advantage(paths)
        else:
            # if false - just use rewards to go as advantages  
            if 'rews_to_go' not in paths.rollouts[0].data.keys():
                paths.add_disc_cumsum('rews_to_go', 'rews',
                                      self.gamma.value(self.global_step))
            advantages = paths.unpack(['rews_to_go'])
            
        # add intristic advantages if needed
        if self.exp_critic is not None and self.weight_i.value(self.global_step) > 0:
            
            advantages_i = self.exp_critic.estimate_advantage(paths)
            
            # combine advantages streams
            weight_i = self.weight_i.value(self.global_step)
            weight_e = self.weight_e.value(self.global_step)
            advantages = weight_i * advantages_i + weight_e * advantages
        
        # standardize advantages for stability    
        if self.standardize_advantages:
            advantages = normalize(advantages, advantages.mean(), advantages.std())
            
        obs = from_numpy(torch.device('cpu'), obs)
        acs = from_numpy(torch.device('cpu'), acs)
        advantages = from_numpy(torch.device('cpu'), advantages)
        
        # evaluate old policy logprob
        logprob_loader = DataLoader(
            list(zip(obs, acs)), 
            batch_size=self.batch_size.value(self.global_step),
            shuffle=False,
            num_workers=self.dataloader_n_workers
        )
        
        fixed_log_prob, fixed_old_policy = [], []
        for obs_i, acs_i in logprob_loader:
            obs_i = obs_i.to(self.device)
            acs_i = acs_i.to(self.device)
            fixed_old_policy_i = detach_dist(self.policy_net(obs_i))
            fixed_log_prob_i = fixed_old_policy_i.log_prob(acs_i).clone()
            fixed_log_prob_i = fixed_log_prob_i.to(torch.device('cpu'))
            fixed_log_prob.append(fixed_log_prob_i)
            fixed_old_policy.append(fixed_old_policy_i)
        fixed_log_prob = torch.cat(fixed_log_prob)
        
        # Create dataloader:
        dataloader = DataLoader(
            list(zip(obs,
                     acs,
                     advantages,
                     fixed_log_prob)), 
            batch_size=self.batch_size.value(self.global_step),
            shuffle=True,
            num_workers=self.dataloader_n_workers
        )
        
        # performing PPO update
        # starting policy training loop
        act_n_epochs = 0
        early_stop = 0
        
        for _ in range(self.n_epochs_per_update.value(self.global_step)):
            
            loss_vec, entropy_vec = [], []

            for obs_i, acs_i, advantages_i, fixed_log_prob_i in dataloader:

                obs_i = obs_i.to(self.device)
                acs_i = acs_i.to(self.device)
                advantages_i = advantages_i.to(self.device)
                fixed_log_prob_i = fixed_log_prob_i.to(self.device)

                loss, entropy_i = self.surrogate_loss(
                    obs=obs_i,
                    acs=acs_i,
                    advantages=advantages_i,
                    fixed_log_prob=fixed_log_prob_i
                )

                # performing gradient step
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_norm_clipping is not None:
                    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_norm_clipping)
                self.optimizer.step()

                # log values
                loss_vec.append(loss.item())
                entropy_vec.append(entropy_i)

            mean_loss = np.mean(loss_vec)
            mean_entropy = np.mean(entropy_vec)
            
            # calculating KL_div
            fixed_new_policy = []
            for obs_i, _ in logprob_loader:
                obs_i = obs_i.to(self.device)
                fixed_new_policy_i = detach_dist(self.policy_net(obs_i))
                fixed_new_policy.append(fixed_new_policy_i)
            
            mean_kl_div = []
            for od, nd in zip(fixed_old_policy, fixed_new_policy):
                kl_div = torch.mean(kl_divergence(od, nd)).item()
                mean_kl_div.append(kl_div)
            mean_kl_div = np.mean(mean_kl_div)
            
            self.n_updates += 1 
            act_n_epochs += 1
            
            # Early stopping if needed:
            if mean_kl_div > self.target_kl.value(self.global_step):
                early_stop += 1
                break
        
        pr = type(self).__name__
        self.last_logs = {f'{pr}_actor_loss': mean_loss,
                          f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                          f'{pr}_gamma': self.gamma.value(self.global_step),
                          f'{pr}_eps': self.eps.value(self.global_step),
                          f'{pr}_ent_coef': self.ent_coef.value(self.global_step),
                          f'{pr}_target_kl': self.target_kl.value(self.global_step),
                          f'{pr}_avg_step_kl_div': mean_kl_div,
                          f'{pr}_act_n_epochs': act_n_epochs,
                          f'{pr}_early_stop': early_stop,
                          f'{pr}_avg_policy_entropy': mean_entropy}
                       
        self.schedules_step()
        
        return self.last_logs     

                  
class ArgmaxQValue(BaseActor, metaclass=abc.ABCMeta):
    
    def __init__(self, 
                 device=torch.device('cpu'),
                 exploration=None):
        
        self.device = device
        self.exploration = exploration
        self.global_step = 0
        self.critic = None
        self.train_sampling = False
        
        self.ckpt_attrs = ['global_step']
        self.last_logs = {}
    
    def set_critic(self, critic):
        
        # pass critic
        self.critic = critic
        
        # pass lag parameters if needed
        if all(item in critic.__dict__.keys() for item in ['obs_nlags', 
                                                           'obs_concat_axis', 
                                                           'obs_expand_axis',
                                                           'obs_padding']):
            self.obs_nlags = critic.obs_nlags
            self.obs_concat_axis = critic.obs_concat_axis
            self.obs_expand_axis = critic.obs_expand_axis
            self.obs_padding = critic.obs_padding
            
    def set_device(self, device):
        self.device = device
    
    def update(self, placeholder=None):
        self.schedules_step()
        self.last_logs = {}
        pr = type(self).__name__
        self.last_logs[f'{pr}_global_step'] = self.global_step
        if self.exploration is not None:
            self.last_logs = {**self.last_logs, **self.exploration.get_logs()}
        return self.last_logs
    
    def schedules_step(self):
        self.global_step += 1
        if self.exploration is not None:
            self.exploration.schedules_step()
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        
        if self.critic is None:
            raise AssertionError(f'Critic has not been provided to {type(self).__name__}')
        else:
            acs = None
            logits = self.critic.forward_np(obs)
            if self.train_sampling:
                if self.exploration is None:
                    acs = logits.argmax(-1)
                else:
                    acs = self.exploration.get_action(logits=logits)
            else:
                acs = logits.argmax(-1)
        return acs.copy()

    
class DDPG(BaseActor,
           DynaAccelerator,
           nn.Module, 
           metaclass=abc.ABCMeta):
    
    def __init__(self,
                 device,
                 mu_net,
                 learning_rate,
                 batch_size,
                 tau=1e-3,
                 n_steps_per_update=1,
                 update_freq=1,
                 target_updates_freq=1,
                 exploration=None,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        from relax.rl.critics import CDQN
        from relax.exploration import OrnsteinUhlenbeck, RandomNormal
        
        self.train_sampling = False
        
        # initialize all scheduled args as schedules
        # if they are already not schedules
        self.global_step = 0
        self.local_step = 0
        self.n_updates = 0
        self.n_target_updates = 0
        self.ckpt_attrs = ['global_step', 'local_step', 'n_updates', 'n_target_updates']
        
        self.learning_rate = init_schedule(learning_rate)
        self.tau = init_schedule(tau)
        self.batch_size = init_schedule(batch_size, discrete=True)
        self.update_freq = init_schedule(update_freq, discrete=True)
        self.target_updates_freq = init_schedule(target_updates_freq, discrete=True)
        self.n_steps_per_update = init_schedule(n_steps_per_update, discrete=True)
        
        # Exploration 
        self.exploration = exploration
        self.valid_expl_types = (OrnsteinUhlenbeck, RandomNormal)
        assert self.exploration is None or isinstance(self.exploration, 
                                                      self.valid_expl_types)
        
        # constant params  
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        # nn.Module params  
        self.device = device
        # mu net
        self.mu_net = mu_net
        self.mu_net.to(self.device)
        
        self.optimizer = optim.Adam(self.mu_net.parameters(), lr=1, eps=1e-6)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lambda t: self.learning_rate.value(t))
        
        # target net
        self.target_mu_net = deepcopy(mu_net)
        self.target_mu_net.to(self.device)
        
        # Critic
        self.critic = None
        self.valid_critic = CDQN
        
        # loging:
        self.last_logs = {}
        
    def set_device(self, device):
        self.device = device
        self.mu_net.to(self.device)
        self.target_mu_net.to(self.device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        
    def set_critic(self, critic):
        
        if not isinstance(critic, self.valid_critic):
            raise ValueError(
                f'{type(self).__name__} only supports {self.valid_critic} as critic'
            )
            
        # pass critic
        self.critic = critic
        
    def schedules_step(self):
        self.global_step += 1
        if hasattr(self, 'scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
        if self.exploration is not None:
            self.exploration.schedules_step()
            
    def forward(self, obs: torch.FloatTensor,
                target=False) -> torch.FloatTensor:
        if target:
            return self.target_mu_net(obs)
        else:
            return self.mu_net(obs)
    
    def forward_np(self, obs: np.ndarray,
                   target=False) -> np.ndarray:
        obs = from_numpy(self.device, obs)
        acs = self.forward(obs=obs, target=target)
        return to_numpy(acs)
    
    def update_target_network(self):
        tau = self.tau.value(self.global_step)
        self.n_target_updates += 1
        for target_param, param in zip(
            self.target_mu_net.parameters(), self.mu_net.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        acs = self.forward_np(obs)
        if self.train_sampling and self.exploration is not None:
            if isinstance(self.exploration, self.valid_expl_types):
                acs = self.exploration.get_action(acs=acs)
            else:
                raise AttributeError(
                    f'{type(self).__name__} only supports {self.valid_expl_types} as exploration'
                )
        return acs
    
    def update(self, buffer: ReplayBuffer) -> dict:
        
        if self.critic is None:
            raise AttributeError(
                f'{type(self).__name__} can not be trained without {self.valid_critic} critic'
            )
        
        # perform training loop
        for _ in range(self.n_steps_per_update.value(self.global_step)):
            
            if (self.learning_rate.value(self.global_step) > 0
                and self.local_step % self.update_freq.value(self.global_step) == 0):

                # sampling self.batch_size transitions:
                batch_size = self.batch_size.value(self.global_step)
                
                # check if prioritization is needed and sample
                sample = buffer.sample(batch_size=batch_size,
                                       p_learner=self.critic if self.critic.prioritized_sampling else None)
                
                # DYNA acceleration if needed
                if hasattr(self, 'acceleration'):
                    sample.accelerate(**self.acceleration_config)
                
                # handling multistep learning and crating next_obs:
                n_steps = self.critic.n_step_learning.value(self.critic.global_step)
                gamma = self.critic.gamma.value(self.critic.global_step)
                rews, gamma_pow, terminals = handle_n_step(data=sample, 
                                                           n=n_steps, 
                                                           gamma=gamma)
                
                # do not recalculate if lag profile is the same
                # create config string and add to lagged variable name
                actor_config = '_'.join([str(l) for l in [self.obs_nlags, 
                                                          self.obs_concat_axis, 
                                                          self.obs_expand_axis, 
                                                          self.obs_padding]])
                
                critic_config = '_'.join([str(l) for l in [self.critic.obs_nlags, 
                                                           self.critic.obs_concat_axis, 
                                                           self.critic.obs_expand_axis, 
                                                           self.critic.obs_padding]])
                
                # Preparing the data for actor
                # creating lags if needed in model
                # unpack rollouts for training
                obs, next_obs = handle_lags(data=sample,
                                            fields={'obs':'lag_concat_obs_' + actor_config,
                                                    'next_obs': 'lag_concat_next_obs_' + actor_config},
                                            nlags=self.obs_nlags,
                                            concat_axis=self.obs_concat_axis,
                                            expand_axis=self.obs_expand_axis,
                                            padding=self.obs_padding)
                
                # Preparing the data for critic
                # creating lags if needed in model
                # unpack rollouts for training
                critic_obs, critic_next_obs = handle_lags(data=sample,
                                                          fields={'obs':'lag_concat_obs_' + critic_config,
                                                                  'next_obs': 'lag_concat_next_obs_' + critic_config},
                                                          nlags=self.critic.obs_nlags,
                                                          concat_axis=self.critic.obs_concat_axis,
                                                          expand_axis=self.critic.obs_expand_axis,
                                                          padding=self.critic.obs_padding)

                acs = sample.unpack(['acs'])
                
                # Pre-calculate is_weights if needed
                is_weights = None
                if self.critic.prioritized_sampling:
                    # prioritized case
                    # compute importance sampling weights MAY BE DO IT INSIDE sample?
                    betta = self.critic.betta.value(self.critic.global_step)
                    
                    p_alpha = sample.unpack(['p_alpha'])
                    
                    N = sample.parent_buffer.n_transitions
                    p_alpha_total = sample.get_priority_sum(p_learner=self.critic)
                    
                    probs = p_alpha / p_alpha_total
                    
                    is_weights = (N * probs)**(-betta)
                    is_weights = is_weights / is_weights.max()
                
                ### Implementing DDPG update ###
                # Continiuos Q-value critic update
                self.critic._ddpg_update(obs=critic_obs,
                                         next_obs=critic_next_obs,
                                         acs=acs, 
                                         rews=rews,
                                         gamma_pow=gamma_pow,
                                         terminals=terminals,
                                         is_weights=is_weights,
                                         actor_next_obs=next_obs,
                                         actor=self,
                                         sample=sample)
                
                # Transfer the data to required device
                obs = from_numpy(self.device, obs)   
                critic_obs = from_numpy(self.critic.device, critic_obs)
                
                # Deterministic policy update
                actor_actions = self.forward(obs=obs)
                
                elementwise_ddpg_loss = - self.critic.forward(
                    obs=critic_obs, 
                    acs=actor_actions.to(self.critic.device)
                )
                elementwise_ddpg_loss.to(self.device)
                
                if self.critic.prioritized_sampling:         
                    # Compute weighted loss
                    is_weights = from_numpy(self.device, is_weights)
                    ddpg_loss = torch.mean(elementwise_ddpg_loss*is_weights)
                else:
                    # Uniform case
                    ddpg_loss = elementwise_ddpg_loss.mean()

                # performing gradient step
                self.optimizer.zero_grad()
                ddpg_loss.backward()
                self.optimizer.step()
                self.n_updates += 1
                
                # update target networks
                if self.n_updates % self.target_updates_freq.value(self.global_step) == 0 and self.n_updates > 0:
                    self.critic.update_target_network()
                    self.update_target_network()
                    
                # logging
                pr = type(self).__name__
                self.last_logs = {f'{pr}_actor_loss': to_numpy(ddpg_loss),
                                  f'{pr}_tau': self.tau.value(self.global_step),
                                  f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                                  f'{pr}_batch_size': sample.n_transitions,
                                  f'{pr}_global_step': self.global_step,
                                  f'{pr}_local_step': self.local_step,
                                  f'{pr}_n_updates': self.n_updates,
                                  f'{pr}_n_target_updates': self.n_target_updates}
                
                if self.exploration is not None:
                    self.last_logs = {**self.last_logs, **self.exploration.get_logs()}
                    
                # delete sample
                del sample

            self.local_step += 1
        
        # global step for schedules
        self.schedules_step()
        
        return self.last_logs
    
    
class TD3(DDPG):
    
    def __init__(self,
                 device,
                 mu_net: nn.Module,
                 learning_rate,
                 batch_size,
                 tps_sigma=0.2,
                 tps_clipping=0.5,
                 min_acs=None,
                 max_acs=None,
                 tau=5e-3,
                 n_steps_per_update=1,
                 update_freq=1,
                 policy_updates_freq=2,
                 target_updates_freq=2,
                 exploration=None,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first',
                 **kwargs):
        
        super().__init__(
            device,
            mu_net,
            learning_rate,
            batch_size,
            tau,
            n_steps_per_update,
            update_freq,
            target_updates_freq,
            exploration,
            obs_nlags,
            obs_concat_axis,
            obs_expand_axis,
            obs_padding,
            **kwargs
        )
        
        self.n_policy_updates = 0
        self.ckpt_attrs.append('n_policy_updates')
        
        self.min_acs = min_acs
        self.max_acs = max_acs
        
        self.tps_sigma = init_schedule(tps_sigma)
        self.tps_clipping = init_schedule(np.abs(tps_clipping))
        self.policy_updates_freq = init_schedule(policy_updates_freq, discrete=True)
        
        self.last_loss = np.nan
    
    def forward_target_smoothed(self,  
                                obs: torch.FloatTensor) -> torch.FloatTensor:
        
        # policy forward pass
        acs = self.forward(obs=obs, target=True)
        
        # creating policy smoothing noise with required properties
        noise = np.random.normal(loc=0,
                                 scale=self.tps_sigma.value(self.global_step),
                                 size=list(acs.shape))
        clipping = self.tps_clipping.value(self.global_step)
        noise = np.clip(noise, a_min=-clipping, a_max=clipping)
        
        # alter policy's actions
        noise = from_numpy(self.device, noise)
        smoothed_acs = acs + noise
        
        # clip them to a valid range
        if self.min_acs is not None or self.max_acs is not None:
            smoothed_acs = torch.clamp(smoothed_acs,
                                       min=self.min_acs,
                                       max=self.max_acs)
            
        return smoothed_acs
    
    def update(self, buffer: ReplayBuffer) -> dict:
        
        if self.critic is None:
            raise AttributeError(
                f'{type(self).__name__} can not be trained without {self.valid_critic} critic'
            )
        
        # perform training loop
        for _ in range(self.n_steps_per_update.value(self.global_step)):
            
            if (self.learning_rate.value(self.global_step) > 0
                and self.local_step % self.update_freq.value(self.global_step) == 0):

                # sampling self.batch_size transitions:
                batch_size = self.batch_size.value(self.global_step)
                
                # check if prioritization is needed and sample
                sample = buffer.sample(batch_size=batch_size,
                                       p_learner=self.critic if self.critic.prioritized_sampling else None)
                
                # DYNA acceleration if needed
                if hasattr(self, 'acceleration'):
                    sample.accelerate(**self.acceleration_config)
                
                # handling multistep learning and crating next_obs:
                n_steps = self.critic.n_step_learning.value(self.critic.global_step)
                gamma = self.critic.gamma.value(self.critic.global_step)
                rews, gamma_pow, terminals = handle_n_step(data=sample, 
                                                           n=n_steps, 
                                                           gamma=gamma)
                
                # do not recalculate if lag profile is the same
                # create config string and add to lagged variable name
                actor_config = '_'.join([str(l) for l in [self.obs_nlags, 
                                                          self.obs_concat_axis, 
                                                          self.obs_expand_axis, 
                                                          self.obs_padding]])
                
                critic_config = '_'.join([str(l) for l in [self.critic.obs_nlags, 
                                                           self.critic.obs_concat_axis, 
                                                           self.critic.obs_expand_axis, 
                                                           self.critic.obs_padding]])
                
                # Preparing the data for actor
                # creating lags if needed in model
                # unpack rollouts for training
                obs, next_obs = handle_lags(data=sample,
                                            fields={'obs':'lag_concat_obs_' + actor_config,
                                                    'next_obs': 'lag_concat_next_obs_' + actor_config},
                                            nlags=self.obs_nlags,
                                            concat_axis=self.obs_concat_axis,
                                            expand_axis=self.obs_expand_axis,
                                            padding=self.obs_padding)
                
                # Preparing the data for critic
                # creating lags if needed in model
                # unpack rollouts for training
                critic_obs, critic_next_obs = handle_lags(data=sample,
                                                          fields={'obs':'lag_concat_obs_' + critic_config,
                                                                  'next_obs': 'lag_concat_next_obs_' + critic_config},
                                                          nlags=self.critic.obs_nlags,
                                                          concat_axis=self.critic.obs_concat_axis,
                                                          expand_axis=self.critic.obs_expand_axis,
                                                          padding=self.critic.obs_padding)

                acs = sample.unpack(['acs'])
                
                # Pre-calculate is_weights if needed
                is_weights = None
                if self.critic.prioritized_sampling:
                    # prioritized case
                    # compute importance sampling weights MAY BE DO IT INSIDE sample?
                    betta = self.critic.betta.value(self.critic.global_step)
                    
                    p_alpha = sample.unpack(['p_alpha'])
                    
                    N = sample.parent_buffer.n_transitions
                    p_alpha_total = sample.get_priority_sum(p_learner=self.critic)
                    
                    probs = p_alpha / p_alpha_total
                    
                    is_weights = (N * probs)**(-betta)
                    is_weights = is_weights / is_weights.max()
                
                ### Implementing TD3 update ###
                # Continiuos Q-value critic update
                self.critic._td3_update(obs=critic_obs,
                                        next_obs=critic_next_obs,
                                        acs=acs, 
                                        rews=rews,
                                        gamma_pow=gamma_pow,
                                        terminals=terminals,
                                        is_weights=is_weights,
                                        actor_next_obs=next_obs,
                                        actor=self,
                                        sample=sample)
                self.n_updates += 1
                
                # Transfer the data to required device
                obs = from_numpy(self.device, obs)   
                critic_obs = from_numpy(self.critic.device, critic_obs)
                
                # Deterministic policy update
                if self.n_updates % self.policy_updates_freq.value(self.global_step) == 0 and self.n_updates > 0:
                    
                    actor_actions = self.forward(obs=obs)
                    elementwise_td3_loss = - self.critic.forward(
                        obs=critic_obs,
                        acs=actor_actions.to(self.critic.device)
                    )
                    elementwise_td3_loss.to(self.device)
                    
                    if self.critic.prioritized_sampling:         
                        # Compute weighted loss
                        is_weights = from_numpy(self.device, is_weights)
                        td3_loss = torch.mean(elementwise_td3_loss*is_weights)
                    else:
                        # Uniform case
                        td3_loss = elementwise_td3_loss.mean()
                    
                    # performing gradient step
                    self.optimizer.zero_grad()
                    td3_loss.backward()
                    self.optimizer.step()
                    self.n_policy_updates += 1
                    self.last_loss = to_numpy(td3_loss)
                
                
                # update target networks
                if self.n_updates % self.target_updates_freq.value(self.global_step) == 0 and self.n_updates > 0:
                    self.critic.update_target_network()
                    self.update_target_network()
                    
                # logging
                pr = type(self).__name__
                self.last_logs = {f'{pr}_actor_loss': self.last_loss,
                                  f'{pr}_tau': self.tau.value(self.global_step),
                                  f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                                  f'{pr}_batch_size': sample.n_transitions,
                                  f'{pr}_global_step': self.global_step,
                                  f'{pr}_local_step': self.local_step,
                                  f'{pr}_n_updates': self.n_updates,
                                  f'{pr}_n_policy_updates': self.n_policy_updates,
                                  f'{pr}_n_target_updates': self.n_target_updates}
                
                if self.exploration is not None:
                    self.last_logs = {**self.last_logs, **self.exploration.get_logs()}
                    
                # delete sample
                del sample

            self.local_step += 1
        
        # global step for schedules
        self.schedules_step()
        
        return self.last_logs
    
    
class SAC(BaseActor,
          DynaAccelerator,
          nn.Module, 
          metaclass=abc.ABCMeta):
    
    def __init__(self,
                 device,
                 policy_net,
                 learning_rate,
                 batch_size,
                 alpha=None,
                 target_entropy=None,
                 auto_tune_alpha=True,
                 n_steps_per_update=1,
                 update_freq=1,
                 target_updates_freq=1,
                 n_random_steps=0,
                 min_acs=None,
                 max_acs=None,
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first',
                 **kwargs):
        
        super().__init__(**kwargs)
        
        from relax.rl.critics import CDQN
        
        self.train_sampling = False
        
        # initialize all scheduled args as schedules
        # if they are already not schedules
        self.global_step = 0
        self.local_step = 0
        self.n_updates = 0
        self.ckpt_attrs = ['global_step', 'local_step', 'n_updates']
        
        self.learning_rate = init_schedule(learning_rate)
        self.batch_size = init_schedule(batch_size, discrete=True)
        self.update_freq = init_schedule(update_freq, discrete=True)
        self.target_updates_freq = init_schedule(target_updates_freq, discrete=True)
        self.n_steps_per_update = init_schedule(n_steps_per_update, discrete=True)
        
        # constant params  
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        self.min_acs = min_acs
        self.max_acs = max_acs
        self.n_random_steps = n_random_steps
        
        if self.n_random_steps is not None and self.n_random_steps > 0:
            if self.min_acs is None or self.max_acs is None:
                raise ValueError(
                    f'Provide both min_acs and max_acs when using n_random_steps > 0'
                )
        
        # nn.Module params  
        self.device = device
        # policy net
        self.policy_net = policy_net
        self.policy_net.to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1, eps=1e-6)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lambda t: self.learning_rate.value(t))
        
        # Initializing temperature
        self.auto_tune_alpha = auto_tune_alpha
        if not self.auto_tune_alpha:
            if alpha is None:
                raise ValueError(
                    "Provide alpha in case if automatic temperature tuning is False"
                )
            self.alpha_schedule = init_schedule(alpha)
            self.alpha = None
        else:
            if target_entropy is None:
                raise ValueError(
                    "Provide target entropy in case if automatic temperature tuning is True. "
                    "Typically - dim(Action Space) is used."
                )
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = None
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1, eps=1e-6)
            self.alpha_scheduler = optim.lr_scheduler.LambdaLR(self.alpha_optimizer,
                                                               lambda t: self.learning_rate.value(t))
        
        # Critic
        self.critic = None
        self.valid_critic = CDQN
        
        # loging:
        self.last_logs = {}
        
    def set_device(self, device):
        self.device = device
        self.policy_net.to(self.device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        if self.auto_tune_alpha:
            self.log_alpha.to(self.device)
            self.alpha_optimizer.load_state_dict(self.alpha_optimizer.state_dict())
        
    def set_critic(self, critic):
        
        if not isinstance(critic, self.valid_critic):
            raise ValueError(
                f'{type(self).__name__} only supports {self.valid_critic} as critic'
            )
            
        # pass critic
        self.critic = critic
        
    def schedules_step(self):
        self.global_step += 1
        if hasattr(self, 'scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
        if hasattr(self, 'alpha_scheduler'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.alpha_scheduler.step()
                     
    def forward(self, obs: torch.FloatTensor) -> torch.distributions.Distribution:
        return self.policy_net(obs)
    
    def forward_np(self, obs: np.ndarray) -> np.ndarray:
        obs = from_numpy(self.device, obs)
        dist = self.forward(obs)
        acs = dist.sample().detach()
        return to_numpy(acs)
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.train_sampling:
            # get actions with stochasticity
            acs = self.forward_np(obs)
            if self.global_step < self.n_random_steps:
                # use exploration
                acs = np.random.uniform(self.min_acs, 
                                        self.max_acs, 
                                        size=acs.shape)    
        else:
            # ignore stochasticity and take means as actions
            dist = self.forward(from_numpy(self.device, obs))
            if hasattr(dist, 'deterministic_acs'):
                acs = dist.deterministic_acs
                acs = to_numpy(acs)
            else:
                acs = self.forward_np(obs)
        return acs
    
    def update(self, buffer: ReplayBuffer) -> dict:
        
        if self.critic is None:
            raise AttributeError(
                f'{type(self).__name__} can not be trained without {self.valid_critic} critic'
            )
        
        # perform training loop
        for _ in range(self.n_steps_per_update.value(self.global_step)):
            
            if (self.learning_rate.value(self.global_step) > 0
                and self.local_step % self.update_freq.value(self.global_step) == 0):

                # sampling self.batch_size transitions:
                batch_size = self.batch_size.value(self.global_step)
                
                # check if prioritization is needed and sample
                sample = buffer.sample(batch_size=batch_size,
                                       p_learner=self.critic if self.critic.prioritized_sampling else None)
                
                # DYNA acceleration if needed
                if hasattr(self, 'acceleration'):
                    sample.accelerate(**self.acceleration_config)
                
                # handling multistep learning and crating next_obs:
                n_steps = self.critic.n_step_learning.value(self.critic.global_step)
                gamma = self.critic.gamma.value(self.critic.global_step)
                rews, gamma_pow, terminals = handle_n_step(data=sample, 
                                                           n=n_steps, 
                                                           gamma=gamma)
                
                # do not recalculate if lag profile is the same
                # create config string and add to lagged variable name
                actor_config = '_'.join([str(l) for l in [self.obs_nlags, 
                                                          self.obs_concat_axis, 
                                                          self.obs_expand_axis, 
                                                          self.obs_padding]])
                
                critic_config = '_'.join([str(l) for l in [self.critic.obs_nlags, 
                                                           self.critic.obs_concat_axis, 
                                                           self.critic.obs_expand_axis, 
                                                           self.critic.obs_padding]])
                
                # Preparing the data for actor
                # creating lags if needed in model
                # unpack rollouts for training
                obs, next_obs = handle_lags(data=sample,
                                            fields={'obs':'lag_concat_obs_' + actor_config,
                                                    'next_obs': 'lag_concat_next_obs_' + actor_config},
                                            nlags=self.obs_nlags,
                                            concat_axis=self.obs_concat_axis,
                                            expand_axis=self.obs_expand_axis,
                                            padding=self.obs_padding)
                
                # Preparing the data for critic
                # creating lags if needed in model
                # unpack rollouts for training
                critic_obs, critic_next_obs = handle_lags(data=sample,
                                                          fields={'obs':'lag_concat_obs_' + critic_config,
                                                                  'next_obs': 'lag_concat_next_obs_' + critic_config},
                                                          nlags=self.critic.obs_nlags,
                                                          concat_axis=self.critic.obs_concat_axis,
                                                          expand_axis=self.critic.obs_expand_axis,
                                                          padding=self.critic.obs_padding)

                acs = sample.unpack(['acs'])
                
                # Pre-calculate is_weights if needed
                is_weights = None
                if self.critic.prioritized_sampling:
                    # prioritized case
                    # compute importance sampling weights MAY BE DO IT INSIDE sample?
                    betta = self.critic.betta.value(self.critic.global_step)
                    
                    p_alpha = sample.unpack(['p_alpha'])
                    
                    N = sample.parent_buffer.n_transitions
                    p_alpha_total = sample.get_priority_sum(p_learner=self.critic)
                    
                    probs = p_alpha / p_alpha_total
                    
                    is_weights = (N * probs)**(-betta)
                    is_weights = is_weights / is_weights.max()
                
                ### Implementing SAC update ###
                # Set temperature for the update
                if not self.auto_tune_alpha:
                    self.alpha = self.alpha_schedule.value(self.global_step)
                else:
                    self.alpha = torch.exp(self.log_alpha).detach().item()
                
                # Continiuos Q-value critic update
                self.critic._sac_update(obs=critic_obs,
                                        next_obs=critic_next_obs,
                                        acs=acs, 
                                        rews=rews,
                                        gamma_pow=gamma_pow,
                                        terminals=terminals,
                                        is_weights=is_weights,
                                        actor_next_obs=next_obs,
                                        actor=self,
                                        sample=sample)
                
                # Transfer the data to required device
                obs = from_numpy(self.device, obs)   
                critic_obs = from_numpy(self.critic.device, critic_obs)
                
                # Stochastic maximum entropy policy update
                # Get policy actions via reparametrization trick
                policy_dist = self.forward(obs=obs)
                actor_actions = policy_dist.rsample()
                
                # Retreive Q-values from critic_net 
                q_value1 = self.critic.forward(
                    obs=critic_obs,
                    acs=actor_actions.to(self.critic.device)
                )
                
                # Retreive Q-values from critic_net2
                q_value2 = self.critic.critic_net2(
                    obs=critic_obs,
                    acs=actor_actions.to(self.critic.device)
                )
                
                # Compute minimal Q-values
                q_value_min = torch.min(q_value1, q_value2).to(self.device)
                
                # Augument with log_prob
                log_prob = policy_dist.log_prob(actor_actions)
                elementwise_sac_loss = self.alpha * log_prob - q_value_min
                
                if self.critic.prioritized_sampling:         
                    # Compute weighted loss
                    is_weights = from_numpy(self.device, is_weights)
                    sac_loss = torch.mean(elementwise_sac_loss*is_weights)
                else:
                    # Uniform case
                    sac_loss = elementwise_sac_loss.mean()
                
                # performing gradient step
                self.optimizer.zero_grad()
                sac_loss.backward()
                self.optimizer.step()
                self.n_updates += 1
                
                # Update temperature
                if self.auto_tune_alpha:
                    
                    elementwise_alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach())
                    
                    if self.critic.prioritized_sampling:         
                        # Compute weighted loss
                        # is weights are already sent to self.device
                        alpha_loss = torch.mean(elementwise_alpha_loss*is_weights)
                    else:
                        # Uniform case
                        alpha_loss = elementwise_alpha_loss.mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                
                # update target networks
                if self.n_updates % self.target_updates_freq.value(self.global_step) == 0 and self.n_updates > 0:
                    self.critic.update_target_network()
                    
                # logging
                pr = type(self).__name__
                self.last_logs = {f'{pr}_actor_loss': to_numpy(sac_loss),
                                  f'{pr}_alpha': self.alpha,
                                  f'{pr}_learning_rate': self.optimizer.param_groups[0]['lr'],
                                  f'{pr}_batch_size': sample.n_transitions,
                                  f'{pr}_global_step': self.global_step,
                                  f'{pr}_local_step': self.local_step,
                                  f'{pr}_n_updates': self.n_updates}
                
                if self.auto_tune_alpha:
                    alpha_logs = {f'{pr}_alpha_loss': alpha_loss.item()}
                    self.last_logs = {**self.last_logs, **alpha_logs}
                    
                # delete sample
                del sample

            self.local_step += 1
        
        # global step for schedules
        self.schedules_step()
        
        return self.last_logs


class RandomShooting(BaseActor):
    
    def __init__(self, 
                 horizon, 
                 n_candidate_sequences,
                 n_random_steps,
                 acs_dim,
                 min_acs,
                 max_acs,
                 device=torch.device('cpu')):
        
        self.device = device
        
        self.horizon = init_schedule(horizon, discrete=True)
        self.n_candidate_sequences = init_schedule(n_candidate_sequences, discrete=True)
        
        self.n_random_steps = n_random_steps
        
        self.acs_dim = acs_dim
        self.min_acs = min_acs
        self.max_acs = max_acs
        
        self.global_step = 0
        self.model = None
        self.train_sampling = False
        
        self.ckpt_attrs = ['global_step']
        self.last_logs = {}
        
    def set_model(self, model):
        
        # pass model
        self.model = model
        
        # pass lag parameters if needed
        if all(item in model.__dict__.keys() for item in ['obs_nlags', 
                                                          'obs_concat_axis', 
                                                          'obs_expand_axis',
                                                          'obs_padding']):
            self.obs_nlags = model.obs_nlags
            self.obs_concat_axis = model.obs_concat_axis
            self.obs_expand_axis = model.obs_expand_axis
            self.obs_padding = model.obs_padding
            
    def set_device(self, device):
        self.device = device
    
    def update(self, placeholder=None):
        self.schedules_step()
        self.last_logs = {}
        pr = type(self).__name__
        self.last_logs[f'{pr}_global_step'] = self.global_step
        return self.last_logs
    
    def schedules_step(self):
        self.global_step += 1
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        
        assert len(obs.shape) >= 2 # ensure observation is in batches
        
        if self.model is None:
            raise AssertionError(f'Model has not been provided to {type(self).__name__}')
        
        out_acs = []
        
        for obs_index in range(obs.shape[0]):
        
            if self.global_step < self.n_random_steps and self.train_sampling:

                acs = np.random.uniform(self.min_acs, self.max_acs,
                                        [self.acs_dim])

            else:

                n_candidate_sequences = self.n_candidate_sequences.value(self.global_step)
                horizon = self.horizon.value(self.global_step)

                # repeat original obs n_candidate_sequences times
                repeated_obs = np.repeat(np.expand_dims(obs[obs_index], axis=0), 
                                         n_candidate_sequences, 
                                         axis=0)

                # generate random action sequences
                candidate_action_sequences = np.random.uniform(self.min_acs, self.max_acs,
                                                               [horizon, n_candidate_sequences, self.acs_dim])

                # evaluate them with a model
                _, pred_rews, terminals = self.model.predict_action_sequence(
                    lag_obs=repeated_obs,
                    action_sequence=candidate_action_sequences
                )
                
                # discard beyond terminal transitions if any
                terminal_mask = terminals.cumsum(axis=0) <= 1
                pred_rews *= terminal_mask

                # choose the best action sequence
                best_action_sequence = candidate_action_sequences[:, pred_rews.sum(axis=0).argmax(), :].copy()

                # return the 1st element of that sequence
                acs = best_action_sequence[0, :].copy()
                
            out_acs.append(acs)
            
        return np.array(out_acs)
        
        
class CEM(RandomShooting):
    
    def __init__(self, 
                 horizon, 
                 n_candidate_sequences,
                 n_iterations,
                 n_elites,
                 alpha,
                 n_random_steps,
                 acs_dim,
                 min_acs,
                 max_acs,
                 device=torch.device('cpu')):
        
        super().__init__(
            horizon=horizon, 
            n_candidate_sequences=n_candidate_sequences,
            n_random_steps=n_random_steps,
            acs_dim=acs_dim,
            min_acs=min_acs,
            max_acs=max_acs,
            device=device
        )
        
        self.alpha = init_schedule(alpha)
        self.n_elites = init_schedule(n_elites, discrete=True)
        self.n_iterations = init_schedule(n_iterations, discrete=True)
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        
        assert len(obs.shape) >= 2 # ensure observation is in batches
        
        if self.model is None:
            raise AssertionError(f'Model has not been provided to {type(self).__name__}')
        
        out_acs = []
        
        n_candidate_sequences = self.n_candidate_sequences.value(self.global_step)
        horizon = self.horizon.value(self.global_step)
        n_iterations = self.horizon.value(self.global_step)
        n_elites = self.n_elites.value(self.global_step)
        alpha = self.alpha.value(self.global_step)
        
        for obs_index in range(obs.shape[0]):
        
            if self.global_step < self.n_random_steps and self.train_sampling:

                acs = np.random.uniform(self.min_acs, self.max_acs,
                                        [self.acs_dim])

            else:

                # repeat original obs n_candidate_sequences times
                repeated_obs = np.repeat(np.expand_dims(obs[obs_index], axis=0), 
                                         n_candidate_sequences, 
                                         axis=0)

                # generate random action sequences
                candidate_action_sequences = np.random.uniform(self.min_acs, self.max_acs,
                                                               [horizon, n_candidate_sequences, self.acs_dim])
                
                # estimate mean random action
                rand_ac_mean = np.mean(candidate_action_sequences, axis=1)
                rand_ac_std = np.std(candidate_action_sequences, axis=1)
                
                # perform CEM sampling loop
                for cem_iter in range(n_iterations):
                    
                    if cem_iter > 0:
                        
                        # sample from a newly fitted distribution
                        # print(horizon, n_candidate_sequences, self.acs_dim)
                        # print(rand_ac_mean.shape, rand_ac_std.shape)
                        candidate_action_sequences = np.random.normal(
                            loc=rand_ac_mean,
                            scale=rand_ac_std,
                            size=[n_candidate_sequences, horizon, self.acs_dim]
                        )
                        
                        # switch to required axis order
                        candidate_action_sequences = np.swapaxes(
                            candidate_action_sequences,
                            axis1=0,
                            axis2=1
                        )
                        
                        # clip to valid range
                        candidate_action_sequences = np.clip(
                            candidate_action_sequences,
                            a_min=self.min_acs,
                            a_max=self.max_acs
                        )
                        
                    # evaluate them with a model
                    _, pred_rews, terminals = self.model.predict_action_sequence(
                        lag_obs=repeated_obs,
                        action_sequence=candidate_action_sequences
                    )
                    
                    # discard beyond terminal transitions if any
                    terminal_mask = terminals.cumsum(axis=0) <= 1
                    pred_rews *= terminal_mask
                    
                    # Choose top self.n_elites sequences
                    top_idxs = (-pred_rews.sum(axis=0)).argsort()[:n_elites]
                    elite_action_sequences = candidate_action_sequences[:, top_idxs, :]

                    # refine mean and std 
                    elite_acs_mean = np.mean(elite_action_sequences, axis=1)
                    rand_ac_mean = alpha * elite_acs_mean + (1 - alpha) * rand_ac_mean

                    elite_acs_std = np.std(elite_action_sequences, axis=1)
                    rand_ac_std = alpha * elite_acs_std + (1 - alpha) * rand_ac_std

                # choose the best action sequence
                best_action_sequence = rand_ac_mean.copy()

                # return the 1st element of that sequence
                acs = best_action_sequence[0, :].copy()
                
            out_acs.append(acs)
            
        return np.array(out_acs)


class FRWR(RandomShooting):
    
    def __init__(self, 
                 horizon, 
                 n_candidate_sequences,
                 n_iterations,
                 gamma,
                 betta,
                 noise_scale,
                 n_random_steps,
                 acs_dim,
                 min_acs,
                 max_acs,
                 device=torch.device('cpu')):
        
        super().__init__(
            horizon=horizon, 
            n_candidate_sequences=n_candidate_sequences,
            n_random_steps=n_random_steps,
            acs_dim=acs_dim,
            min_acs=min_acs,
            max_acs=max_acs,
            device=device
        )
        
        self.gamma = init_schedule(gamma)
        self.betta = init_schedule(betta)
        self.noise_scale = init_schedule(noise_scale)
        self.n_iterations = init_schedule(n_iterations, discrete=True)
        
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        
        assert len(obs.shape) >= 2 # ensure observation is in batches
        
        if self.model is None:
            raise AssertionError(f'Model has not been provided to {type(self).__name__}')
        
        out_acs = []
        
        # check the performance with no schedules
        n_candidate_sequences = self.n_candidate_sequences.value(self.global_step)
        horizon = self.horizon.value(self.global_step)
        n_iterations = self.horizon.value(self.global_step)
        gamma = self.gamma.value(self.global_step)
        betta = self.betta.value(self.global_step)
        noise_scale = self.noise_scale.value(self.global_step)
           
        for obs_index in range(obs.shape[0]):
        
            if self.global_step < self.n_random_steps and self.train_sampling:

                acs = np.random.uniform(self.min_acs, self.max_acs,
                                        [self.acs_dim])

            else:

                # repeat original obs n_candidate_sequences times
                repeated_obs = np.repeat(np.expand_dims(obs[obs_index], axis=0), 
                                         n_candidate_sequences, 
                                         axis=0)

                # generate random action sequences
                candidate_action_sequences = np.random.uniform(self.min_acs, self.max_acs,
                                                               [horizon, n_candidate_sequences, self.acs_dim])
                
                # perform FRWR sampling loop
                for frwr_iter in range(n_iterations):
                    
                    if frwr_iter > 0:
                        
                        # sample noise from normal distribution
                        noise = np.random.normal(
                            loc=np.zeros(self.acs_dim),
                            scale=np.ones(self.acs_dim)*noise_scale,
                            size=[horizon, n_candidate_sequences, self.acs_dim]
                        )
                        
                        # add zero padding
                        padding = np.zeros_like(noise[[0]])
                        noise = np.concatenate([padding, noise], axis=0)
                        
                        # filter the noise
                        filtered_noise = list(
                            accumulate(noise, func=lambda a, b: (1-betta)*a+betta*b)
                        )[-horizon:]
                        filtered_noise = np.array(filtered_noise)
                        
                        # construct action sequrnces
                        candidate_action_sequences = mu[:, np.newaxis, :] + filtered_noise
                        candidate_action_sequences = np.clip(
                            candidate_action_sequences,
                            a_min=self.min_acs,
                            a_max=self.max_acs
                        )
                        
                    # evaluate them with a model
                    _, pred_rews, terminals = self.model.predict_action_sequence(
                        lag_obs=repeated_obs,
                        action_sequence=candidate_action_sequences
                    )
                    
                    # discard beyond terminal transitions if any
                    terminal_mask = terminals.cumsum(axis=0) <= 1
                    pred_rews *= terminal_mask
                    
                    # or subtract the best reward by each timestep TESTING
                    pred_rews = pred_rews.sum(axis=0)
                    pred_rews = normalize(data=pred_rews, mean=pred_rews.mean(), std=pred_rews.std())
                    pred_rews = pred_rews - pred_rews.max()
                    
                    # weight the actions by exp(rews * gamma)
                    exp_rews = np.exp(pred_rews * gamma)
                    exp_rews = np.expand_dims(exp_rews, axis=(0, -1))
                    
                    acs_exp_rews = candidate_action_sequences * exp_rews
                    mu = acs_exp_rews.sum(axis=1) / exp_rews.sum(axis=1)
                    
                # choose the best action sequence
                best_action_sequence = mu.copy()

                # return the 1st element of that sequence
                acs = best_action_sequence[0, :].copy()
                
            out_acs.append(acs)
            
        return np.array(out_acs)
    
