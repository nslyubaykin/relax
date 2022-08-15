import abc
import torch
import warnings
import numpy as np

from torch import nn
from torch import optim
from torch.nn import functional as F

from relax.data.replay_buffer import ReplayBuffer
from relax.data.utils import *
from relax.schedules import init_schedule
from relax.torch.utils import *


class BaseModel(Checkpointer):
    
    def forward_np(self, lag_obs: np.ndarray, acs: np.ndarray) -> tuple:
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError


class DeltaEnvModel(BaseModel, nn.Module):
    
    def __init__(self, 
                 obs_models,
                 device,
                 learning_rate,
                 batch_size,
                 noise=0.0,
                 rews_models=None,
                 reward_function=None,
                 terminal_function=None,
                 n_steps_per_update=1,
                 update_freq=1,
                 stats_recalc_freq=1,
                 weight_decay=0.0,
                 disc_actions=False,
                 reset_models_freq=None,
                 reset_n_models=0,
                 min_updates_since_reset=0,
                 random_agg=False,
                 obs_postproc_function=None,
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
                           'buffer_stats', 'n_model_resets', 'updates_per_model']
        
        self.learning_rate = init_schedule(learning_rate)
        self.noise = init_schedule(noise)
        self.batch_size = init_schedule(batch_size, discrete=True)
        self.update_freq = init_schedule(update_freq, discrete=True)
        self.stats_recalc_freq = init_schedule(stats_recalc_freq, discrete=True)
        self.n_steps_per_update = init_schedule(n_steps_per_update, discrete=True)
        
        if reset_models_freq is None:
            self.reset_models_freq = reset_models_freq
        else:
            self.reset_models_freq = init_schedule(reset_models_freq, discrete=True)
        self.reset_n_models = init_schedule(reset_n_models, discrete=True)
        self.already_reset = []
        self.min_updates_since_reset = min_updates_since_reset
        
        self.random_agg = random_agg
        self.disc_actions = disc_actions
        
        self.obs_postproc_function = obs_postproc_function
        
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        # nn.Module params  
        self.device = device
        self.weight_decay = weight_decay
        
        # obs models
        self.obs_models = []
        self.updates_per_model = []
        self.obs_optimizers = []
        self.obs_schedulers = []
        
        for i, obs_model in enumerate(obs_models):
            
            obs_model.to(self.device)
            obs_optimizer = optim.Adam(obs_model.parameters(), 
                                       lr=1, 
                                       eps=1e-6,
                                       weight_decay=self.weight_decay)
            
            obs_scheduler = optim.lr_scheduler.LambdaLR(obs_optimizer,
                                                        lambda t: self.learning_rate.value(t))
            
            om_name, oo_name, os_name = f'obs_model_{i}', f'obs_optimizer_{i}', f'obs_scheduler_{i}'
            
            # save names
            self.obs_models.append(om_name)
            self.obs_optimizers.append(oo_name)
            self.obs_schedulers.append(os_name)
            self.updates_per_model.append(0)
            
            # set as attributes
            setattr(self, om_name, obs_model)
            setattr(self, oo_name, obs_optimizer)
            setattr(self, os_name, obs_scheduler)
            
        # rews models
        if reward_function is None and rews_models is None:
            raise ValueError(
                f'If rewards models are not provided, provide reward_function arg to query it'
            )
        
        self.rews_models = None
        self.reward_function = reward_function
        
        if rews_models is not None:
            
            assert len(rews_models) == len(self.obs_models)
            
            self.rews_models = []
            self.rews_optimizers = []
            self.rews_schedulers = []
            
            for i, rews_model in enumerate(rews_models):
                
                rews_model.to(self.device)
                rews_optimizer = optim.Adam(rews_model.parameters(), 
                                            lr=1, 
                                            eps=1e-6,
                                            weight_decay=self.weight_decay)

                rews_scheduler = optim.lr_scheduler.LambdaLR(rews_optimizer,
                                                             lambda t: self.learning_rate.value(t))
                
                rm_name, ro_name, rs_name = f'rews_model_{i}', f'rews_optimizer_{i}', f'rews_scheduler_{i}'
            
                # save names
                self.rews_models.append(rm_name)
                self.rews_optimizers.append(ro_name)
                self.rews_schedulers.append(rs_name)
            
                # set as attributes
                setattr(self, rm_name, rews_model)
                setattr(self, ro_name, rews_optimizer)
                setattr(self, rs_name, rews_scheduler)
                
        # terminal function
        self.terminal_function = terminal_function
                
        # loging:
        self.last_logs = {}
        
        # buffer stats
        self.buffer_stats = {}
        
    def set_device(self, device):
        
        self.device = device
        
        # moving obs models and optimizers in a loop
        for om_name, oo_name in zip(self.obs_models, self.obs_optimizers):
            om, oo = getattr(self, om_name), getattr(self, oo_name)
            om.to(self.device)
            oo.load_state_dict(oo.state_dict())
        
        # moving rews models and optimizers in a loop if needed
        if self.rews_models is not None:
            for rm_name, ro_name in zip(self.rews_models, self.rews_optimizers):
                rm, ro = getattr(self, om_name), getattr(self, oo_name)
                rm.to(self.device)
                ro.load_state_dict(ro.state_dict())
                
    def set_reward_function(self, reward_function=None):
        
        if reward_function is None and self.rews_models is None:
            raise ValueError(
                f'If rewards models are not provided, provide reward_function arg to query it'
            )
            
        self.reward_function = reward_function
        
    def schedules_step(self):
        self.global_step += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for os_name in self.obs_schedulers:
                os = getattr(self, os_name)
                os.step()
            if self.rews_models is not None:
                for rs_name in self.rews_schedulers:
                    rs = getattr(self, rs_name)
                    rs.step()
                
    def update_buffer_stats(self, buffer: ReplayBuffer):
        
        buffer.add_next_obs()
        obs, next_obs = buffer.unpack(['obs', 'next_obs'])
        lag_obs = handle_lags(data=buffer,
                              fields={'obs': 'lag_concat_obs'},
                              nlags=self.obs_nlags,
                              concat_axis=self.obs_concat_axis,
                              expand_axis=self.obs_expand_axis,
                              padding=self.obs_padding)
        buffer.drop_field('next_obs')
        buffer.drop_field('lag_concat_obs')

        self.buffer_stats = {
            'lag_obs_mean': np.mean(lag_obs, axis=0),
            'lag_obs_std': np.std(lag_obs, axis=0),
            'deltas_mean': np.mean(next_obs - obs, axis=0),
            'deltas_std': np.std(next_obs - obs, axis=0),
        }
        
        if not self.disc_actions:
            acs = buffer.unpack(['acs'])
            self.buffer_stats['acs_mean'] = np.mean(acs, axis=0)
            self.buffer_stats['acs_std'] = np.std(acs, axis=0)

        if self.rews_models is not None:
            rews = buffer.unpack(['rews'])
            self.buffer_stats['rews_mean'] = np.mean(rews, axis=0)
            self.buffer_stats['rews_std'] = np.std(rews, axis=0)
            
        self.n_stats_updates += 1
        
    def reset_models(self):
        
        # reset only after the start of model training
        if sum(self.updates_per_model) > 0:
        
            # which models stil require reset
            to_reset = [ind for ind in range(len(self.obs_models)) if ind not in self.already_reset]

            # randomly choose models to reset it this iteration
            #iter_reset = np.random.choice(to_reset, 
            #                              min(len(to_reset), self.reset_n_models.value(self.global_step)), 
            #                              replace=False).tolist()
            
            # sequentially choose models to reset it this iteration
            iter_reset = to_reset[:self.reset_n_models.value(self.global_step)]

            # reset required models in a loop
            for i in iter_reset:

                # obs
                # reset model weights
                obs_model = getattr(self, self.obs_models[i])
                obs_model.apply(weight_reset)

                # reset optimizer
                obs_optimizer = optim.Adam(obs_model.parameters(), 
                                           lr=1, 
                                           eps=1e-6,
                                           weight_decay=self.weight_decay)
                setattr(self, self.obs_optimizers[i], obs_optimizer)

                # reset scheduler
                obs_scheduler = optim.lr_scheduler.LambdaLR(obs_optimizer,
                                                            lambda t: self.learning_rate.value(t))
                setattr(self, self.obs_schedulers[i], obs_scheduler)

                # reset updates counter vector
                self.updates_per_model[i] = 0

                # rews if needed
                if self.rews_models is not None:

                    # reset model weights
                    rews_model = getattr(self, self.rews_models[i])
                    rews_model.apply(weight_reset)

                    # reset optimizer
                    rews_optimizer = optim.Adam(rews_model.parameters(), 
                                                lr=1, 
                                                eps=1e-6,
                                                weight_decay=self.weight_decay)
                    setattr(self, self.rews_optimizers[i], rews_optimizer)

                    # reset scheduler
                    rews_scheduler = optim.lr_scheduler.LambdaLR(rews_optimizer,
                                                                 lambda t: self.learning_rate.value(t))
                    setattr(self, self.rews_schedulers[i], rews_scheduler)

            # update the list of models which are already reset
            self.already_reset.extend(iter_reset)

            # reset this list in case if all models were reset already
            if len(self.already_reset) == len(self.obs_models):
                self.already_reset = []

            self.n_model_resets += 1
        
    def get_valid_models(self):
        
        min_upd = self.min_updates_since_reset
        valid_models = [m_ind for m_ind, n_upd in enumerate(self.updates_per_model) if n_upd >= min_upd]
        
        return valid_models

    def forward_np(self, lag_obs: np.ndarray, acs: np.ndarray) -> tuple:
        
        # reconstruct original non-lagged observation
        obs = get_last_lag(lag_obs=lag_obs, nlags=self.obs_nlags,
                           concat_axis=self.obs_concat_axis, expand_axis=self.obs_expand_axis)
        
        # scale lag_obs and acs
        lag_obs_norm = normalize(data=lag_obs,
                                 mean=self.buffer_stats['lag_obs_mean'],
                                 std=self.buffer_stats['lag_obs_std'])
        acs_norm = normalize(data=acs,
                             mean=self.buffer_stats['acs_mean'],
                             std=self.buffer_stats['acs_std'])

        # convert to tensors
        lag_obs_norm = from_numpy(self.device, lag_obs_norm)
        acs_norm = from_numpy(self.device, acs_norm)

        next_obs, rews = [], []
        
        # select models that are old enough to vote
        mature_models = self.get_valid_models()
        assert len(mature_models) > 0
        
        if self.random_agg:
            # choose one random model to vote
            loop_iter = np.random.choice(mature_models, 1).tolist()
        else:
            # make everybody vote
            loop_iter = mature_models

        for i in loop_iter:
            
            # obs
            obs_model = getattr(self, self.obs_models[i])
            
            # models forward pass
            deltas_norm_i = obs_model(lag_obs_norm, acs_norm)
            deltas_norm_i = to_numpy(deltas_norm_i)

            # unnormalize deltas 
            deltas_i = unnormalize(data=deltas_norm_i,
                                   mean=self.buffer_stats['deltas_mean'],
                                   std=self.buffer_stats['deltas_std'])
            

            # calculate next obs
            next_obs_i = obs + deltas_i
            
            # accumulate ensemble data
            next_obs.append(np.expand_dims(next_obs_i, axis=0))
            
            # rews
            if self.reward_function is None:
                
                # use fitted reward function
                rews_model = getattr(self, self.rews_models[i])
                
                # models forward pass
                rews_norm_i = rews_model(lag_obs_norm, acs_norm).squeeze()
                rews_norm_i = to_numpy(rews_norm_i)
                
                # unnormalize rews
                rews_i = unnormalize(data=rews_norm_i,
                                     mean=self.buffer_stats['rews_mean'],
                                     std=self.buffer_stats['rews_std'])
                
                # accumulate ensemble data
                rews.append(np.expand_dims(rews_i, axis=0))
            
        # unify data from ensemble
        next_obs = np.concatenate(next_obs, axis=0)
        
        # aggregate the data from all models
        next_obs = np.mean(next_obs, axis=0)
        
        # post-process obs if needed
        if self.obs_postproc_function is not None:
            next_obs = self.obs_postproc_function(next_obs)
        
        if self.reward_function is None:
            # if no true reward function provided, use fitted rewards
            rews = np.concatenate(rews, axis=0)
            rews = np.mean(rews, axis=0)
        else:
            # othervise query provided reward function
            rews = self.reward_function(lag_obs, acs) # reward function follows general lag profile
            
        # calculating terminals if needed
        if self.terminal_function is not None:
            # Checking if the next transition is terminal
            terminals = self.terminal_function(next_obs)
        else:
            # Assume no terminals otherwise
            terminals = np.zeros_like(rews, dtype=bool)
        
        return next_obs, rews, terminals
    
    def predict_action_sequence(self, lag_obs: np.ndarray, action_sequence: np.ndarray) -> tuple:
        
        # reconstruct original non-lagged observation
        obs = get_last_lag(lag_obs=lag_obs, nlags=self.obs_nlags,
                           concat_axis=self.obs_concat_axis, expand_axis=self.obs_expand_axis)
        
        # init outputs
        pred_obs, pred_rews, pred_terminals = [], [], []
        
        # iteratively predict states and rewards over the sequence:
        for acs in action_sequence:
            
            next_obs, rews, terminals = self.forward_np(lag_obs=lag_obs, acs=acs)
            
            # calculate next lag obs for iterative input
            lag_obs_next = get_next_lag_obs(lag_obs=lag_obs, next_obs=next_obs, 
                                            nlags=self.obs_nlags, concat_axis=self.obs_concat_axis, 
                                            expand_axis=self.obs_expand_axis)

            # save current step predictions
            pred_obs.append(obs)
            pred_rews.append(rews)
            pred_terminals.append(terminals)

            # update values for the next step
            lag_obs = lag_obs_next
            obs = next_obs
        
        # Formulate resulting tensors
        pred_obs, pred_rews, pred_terminals = np.array(pred_obs), np.array(pred_rews), np.array(pred_terminals)
        
        return pred_obs, pred_rews, pred_terminals
            
    def update(self, buffer: ReplayBuffer) -> dict:
        
        # perform training loop
        for _ in range(self.n_steps_per_update.value(self.global_step)):
            
            # update buffer stats is needed
            if (len(self.buffer_stats.keys()) == 0
                or self.local_step % self.stats_recalc_freq.value(self.global_step) == 0):
                
                self.update_buffer_stats(buffer=buffer)
                
            # reset models if needed
            if (self.reset_models_freq is not None
                and self.local_step > 0
                and self.local_step % self.reset_models_freq.value(self.global_step) == 0):
                
                self.reset_models()
                
            if (self.learning_rate.value(self.global_step) > 0
                and self.local_step % self.update_freq.value(self.global_step) == 0):
                
                # Train each model in the ensemble in a loop
                # obs models
                delta_losses = []
                next_obs_losses = []
                rews_losses = []
                
                for i in range(len(self.obs_models)):
                    
                    # Predicting next obs
                    obs_model = getattr(self, self.obs_models[i])
                    obs_optimizer = getattr(self, self.obs_optimizers[i])
                    
                    # sampling self.batch_size transitions: TRY SAMPLING OUTSIDE OF THE LOOP!!
                    batch_size = self.batch_size.value(self.global_step)
                    sample = buffer.sample(batch_size=batch_size) # use different batch for each model
                    
                    # querying 'next_obs' field:
                    sample.add_next_obs()

                    # creating lags if needed in model
                    # unpack rollouts for training
                    acs, obs, next_obs = sample.unpack(['acs', 'obs', 'next_obs'])
                    lag_obs = handle_lags(data=sample,
                                          fields={'obs': 'lag_concat_obs'},
                                          nlags=self.obs_nlags,
                                          concat_axis=self.obs_concat_axis,
                                          expand_axis=self.obs_expand_axis,
                                          padding=self.obs_padding)
                    deltas = next_obs - obs
                    
                    # add noise if needed:
                    noise = self.noise.value(self.global_step)
                    
                    if noise > 0:
                        
                        lag_obs = add_noise(data=lag_obs,
                                            mean_data=self.buffer_stats['lag_obs_mean'],
                                            noise=noise)
                        
                        if not self.disc_actions:
                            acs = add_noise(data=acs,
                                            mean_data=self.buffer_stats['acs_mean'],
                                            noise=noise)
                            
                        deltas = add_noise(data=deltas,
                                           mean_data=self.buffer_stats['deltas_mean'],
                                           noise=noise)
                    
                    # normalize lag_obs, acs and deltas
                    lag_obs_norm = normalize(data=lag_obs,
                                             mean=self.buffer_stats['lag_obs_mean'],
                                             std=self.buffer_stats['lag_obs_std'])
                    
                    if not self.disc_actions:
                        acs_norm = normalize(data=acs,
                                             mean=self.buffer_stats['acs_mean'],
                                             std=self.buffer_stats['acs_std'])
                    else:
                        acs_norm = acs

                    deltas_norm = normalize(data=deltas,
                                            mean=self.buffer_stats['deltas_mean'],
                                            std=self.buffer_stats['deltas_std'])

                    lag_obs_norm = from_numpy(self.device, lag_obs_norm)
                    acs_norm = from_numpy(self.device, acs_norm)
                    deltas_norm = from_numpy(self.device, deltas_norm)
                    next_obs = from_numpy(self.device, next_obs)
                    
                    # training the model
                    deltas_model_norm = obs_model(lag_obs_norm, acs_norm)
                    delta_loss = F.mse_loss(deltas_model_norm, deltas_norm)
                    
                    # Performing gradient step
                    obs_optimizer.zero_grad()
                    delta_loss.backward()
                    obs_optimizer.step()
                    
                    # saving undividual update counter info
                    self.updates_per_model[i] += 1
                    
                    # calculation actual next_obs prediction loss
                    deltas_model_norm = to_numpy(deltas_model_norm)
                    deltas_model = unnormalize(data=deltas_model_norm,
                                               mean=self.buffer_stats['deltas_mean'],
                                               std=self.buffer_stats['deltas_std'])
                    next_obs_model = from_numpy(self.device, obs + deltas_model)
                    next_obs_loss = F.mse_loss(next_obs_model, next_obs)
                    
                    # save loss values
                    delta_losses.append(delta_loss.item())
                    next_obs_losses.append(next_obs_loss.item())
                    
                    # Predicting rewards:
                    if self.rews_models is not None:
                        
                        # unpack rews for training
                        rews = sample.unpack(['rews'])
                        
                        rews_model = getattr(self, self.rews_models[i])
                        rews_optimizer = getattr(self, self.rews_optimizers[i])
                        
                        # adding noise if needed
                        if noise > 0:
                            rews = add_noise(data=rews,
                                             mean_data=self.buffer_stats['rews_mean'],
                                             noise=noise)
                            
                        # normalize rewards
                        rews_norm = normalize(data=rews,
                                              mean=self.buffer_stats['rews_mean'],
                                              std=self.buffer_stats['rews_std'])
                        
                        rews_norm = from_numpy(self.device, rews_norm)
                        
                        # training the model
                        rews_model_norm = rews_model(lag_obs_norm, acs_norm).squeeze()
                        rews_loss = F.mse_loss(rews_model_norm, rews_norm)
                        
                        # Performing gradient step
                        rews_optimizer.zero_grad()
                        rews_loss.backward()
                        rews_optimizer.step()
                        
                        # save loss values
                        rews_losses.append(rews_loss.item())
                        
                    # delete sample
                    del sample
                    
                self.n_updates += 1
                
                # logging
                pr = type(self).__name__
                mature_models = self.get_valid_models()
                self.last_logs = {f'{pr}_delta_loss': np.mean(np.array(delta_losses)[mature_models]),
                                  f'{pr}_next_obs_loss': np.mean(np.array(next_obs_losses)[mature_models]),
                                  f'{pr}_mean_learning_rate': np.mean(
                                      [getattr(self, oo_name).param_groups[0]['lr'] for oo_name in self.obs_optimizers]
                                  ),
                                  f'{pr}_batch_size': batch_size,
                                  f'{pr}_global_step': self.global_step,
                                  f'{pr}_local_step': self.local_step,
                                  f'{pr}_n_updates': self.n_updates,
                                  f'{pr}_n_stats_updates': self.n_stats_updates,
                                  f'{pr}_n_model_resets': self.n_model_resets,
                                  f'{pr}_models_mean_age': np.mean(np.array(self.updates_per_model)[mature_models])}
                
                if self.rews_models is not None:
                    self.last_logs[f'{pr}_rews_loss'] = np.mean(np.array(rews_losses)[mature_models])
            
            self.local_step += 1
        
        # global step for schedules
        self.schedules_step()
        
        return self.last_logs
