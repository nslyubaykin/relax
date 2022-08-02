import ray
import torch

import numpy as np
import multiprocessing as mp

from math import ceil
from copy import deepcopy
from warnings import warn
from itertools import chain

from relax.data.utils import disc_cumsum
from relax.data.acceleration import _pre_proc_data, _accelerate


class Path:
    
    """
    Class for storing environment rollout
    And some useful methods for environmet's path
    """
    
    def __init__(self):
        
        self.data = {
            'obs': [],
            'acs': [],
            'rews': [],
            'terminals': [],
            'is_time_limit': [],
        }
        
        self.steps = 0
        self.has_start = None
        self.has_end = None
        self.to_be_extended = None
        self.envhash = None
        self.terminal_ob = None
        # self.terminal_lag_ob = None
        self.is_real = True
        
        self.next_obs_n = None
        
        self.n_step_rews_n = None
        self.n_step_rews_gamma = None
        
        self.n_gamma_pow_n = None
        self.n_gamma_pow_gamma = None
        
        self.n_step_terminals_n = None
    
    def total_reward(self):
        return sum(self.data['rews'])
    
    def disc_cumsum(self, field: str, gamma: float):
        
        list_of_discounted_cumsums = np.array([])
        
        if self.steps > 0:
            
            list_init = self.data[field]     
            list_of_discounted_cumsums = disc_cumsum(list_init, gamma)
            
        else:
            warn('Path has not been sampled, returning empty np.array',
                 UserWarning)
            
        return list_of_discounted_cumsums
    
    def lag_concat(self, field: str, nlags: int, 
                   concat_axis=-1, expand_axis=None,
                   padding='first'):
        
        lags = []
        
        if self.steps > 0:
            
            flat_lags = []
            pad_obs = []
            data_list = self.data[field]
            
            if expand_axis is not None:
                data_list = [np.expand_dims(data_list_i, axis=expand_axis) for data_list_i in data_list]
                
            flat_lags.append(data_list)

            for lag in range(1, nlags+1):
                
                if field != 'next_obs':
                    pad_ob = self.get(field=field, index=-lag, padding=padding)
                else: # special case for low padding of the next obs
                    pad_ob = self.get(field='obs', index=-lag+self.next_obs_n, padding=padding)
                
                if expand_axis is not None:
                    pad_ob = np.expand_dims(pad_ob, axis=expand_axis)
                    
                pad_obs.append(pad_ob)
                
                lag_data_list = pad_obs[::-1] + data_list[:-lag]
                flat_lags.append(lag_data_list)

            lags = list(map(lambda ll: np.concatenate(ll, axis=concat_axis), zip(*flat_lags[::-1])))
            
            # pad next_obs with valid lagged terminal observation
            if field == 'next_obs':
                pad_n = min(self.steps, self.next_obs_n)
                lags[-pad_n:] = [self._terminal_lag_ob(
                    nlags=nlags, 
                    concat_axis=concat_axis, 
                    expand_axis=expand_axis,
                    padding=padding
                )] * pad_n
                
        else:
            
            warn('Path has not been sampled, returning empty list',
                 UserWarning)
        
        return lags

    def next_obs(self, n=1) -> list:
        
        next_obs = []
        
        if self.steps > 0:
            if self.steps > n:
                next_obs += self.data['obs'][n:]
            next_obs += [self.get(field='obs', index=self.steps+1)] * min(n, self.steps)
        else:
            warn('Path has not been sampled, returning empty list',
                 UserWarning)
        
        return next_obs
    
    def n_step_returns(self, n: int, gamma: float) -> list:

        rews_leads = []
        
        if self.steps > 0:
            
            rews_leads.append(self.data['rews'])
            
            for lead in range(1, n):
                rews_lead = []
                if self.steps > lead:
                    rews_lead += self.data['rews'][lead:]
                rews_lead += [self.get(field='rews', index=self.steps+1)] * min(lead, self.steps)
                rews_leads.append(rews_lead)
                
            rews_leads = np.array(rews_leads)
            rews_leads = (gamma**np.arange(n)[:, np.newaxis] * rews_leads).sum(axis=0).tolist()
            
        else:
            warn('Path has not been sampled, returning empty list',
                 UserWarning)          
        
        return rews_leads
    
    def n_gamma_pow(self, n: int, gamma: float) -> list:
        return (gamma**np.minimum(np.arange(self.steps)[::-1] + 1, n)).tolist()
    
    def n_step_terminals(self, n: int) -> list:
        
        n -= 1 # ternimals already have a 1t shift 
        next_terminals = []
        
        if self.steps > 0:
            if self.steps > n:
                next_terminals += self.data['terminals'][n:]
            next_terminals += [self.get(field='terminals', index=self.steps+1)] * min(n, self.steps)
        else:
            warn('Path has not been sampled, returning empty list',
                 UserWarning)
        
        return next_terminals
    
    def add_disc_cumsum(self, cumsum_field: str, field: str, gamma: float):
        self.data[cumsum_field] = self.disc_cumsum(field=field,
                                                   gamma=gamma)
        
    def add_lag_concat(self, lag_field: str, field: str, nlags: int, 
                       concat_axis=-1, expand_axis=None, padding='first'):
        self.data[lag_field] = self.lag_concat(field=field,
                                               nlags=nlags, 
                                               concat_axis=concat_axis,
                                               expand_axis=expand_axis,
                                               padding=padding)
        
    def add_next_obs(self, n=1):
        if 'next_obs' not in self.data.keys() or self.next_obs_n != n:
            self.next_obs_n = n
            self.data['next_obs'] = self.next_obs(n=n)
            
    def add_n_step_returns(self, n: int, gamma: float):
        if 'n_step_rews' not in self.data.keys() or self.n_step_rews_n != n or self.n_step_rews_gamma != gamma:
            self.n_step_rews_n = n
            self.n_step_rews_gamma = gamma
            self.data['n_step_rews'] = self.n_step_returns(n=n, gamma=gamma)
            
    def add_n_gamma_pow(self, n: int, gamma: float):
        if 'n_gamma_pow' not in self.data.keys() or self.n_gamma_pow_n != n or self.n_gamma_pow_gamma != gamma:
            self.n_gamma_pow_n = n
            self.n_gamma_pow_gamma = gamma
            self.data['n_gamma_pow'] = self.n_gamma_pow(n=n, gamma=gamma)
            
    def add_n_step_terminals(self, n: int):
        if 'n_step_terminals' not in self.data.keys() or self.n_step_terminals_n != n:
            self.n_step_terminals_n = n
            self.data['n_step_terminals'] = self.n_step_terminals(n=n)
        
    def drop_field(self, field: str):
        if field in self.data.keys():
            del self.data[field]
            
    def _terminal_lag_ob(self, nlags: int, 
                         concat_axis=-1, 
                         expand_axis=None,
                         padding='first') -> np.ndarray:
        
        terminal_lag_ob_vec = self.slice_get(
            field='obs', 
            index=(self.steps-nlags, self.steps+1),
            padding=padding
        )

        if expand_axis is not None:
            terminal_lag_ob_vec = [np.expand_dims(t_lag_ob, axis=expand_axis) for t_lag_ob in terminal_lag_ob_vec]

        terminal_lag_ob = np.concatenate(terminal_lag_ob_vec, axis=concat_axis)
        
        return terminal_lag_ob
            
    def _low_pad(self, field: str, index: int, padding='first'):
        
        assert index < 0
        
        out = None
        
        if padding == 'first':
            out = self.data[field][0]
        elif padding == 'zeros':
            out = np.zeros_like(self.data[field][0])
        elif padding == 'debug': # added to debug add_lags_concat method
            out = np.ones_like(self.data[field][0]) * index 
        else:
            raise ValueError(f'{padding} padding is not supported select one of: "first", "zeros"')
            
        return out
    
    def _slice_low_pad(self, field: str, index: tuple, padding='first'):
        
        low, high = index
        
        assert low < 0 and high <= 0
        
        return [self._low_pad(field=field, index=-1, padding=padding)] * len(range(low, high))
        
    def _high_pad(self, field: str, index: int):
        
        assert index >= self.steps
        
        out = None
        
        if field == 'obs':
            out = self.terminal_ob
        elif field == 'rews':
            out = 0
        elif field == 'terminals':
            out = self.data[field][-1]
        else:
            raise IndexError(
                f' _high_pad() is not implemented for {field}. Index {index} is out of range.'
            )
            
        return out
    
    def get(self, field: str, index: int, padding='first'):
        
        out = None
        
        if index >= 0:
            if index < self.steps:
                out = self.data[field][index]
            else:
                out = self._high_pad(field=field, index=index)
        else:
            out = self._low_pad(field=field, index=index, padding=padding)
                   
        return out
    
    def slice_get(self, field: str, index: tuple, padding='first'):
        
        low, high = index

        if low is None:
            low = 0

        if high is None:
            high = self.steps # self.steps

        out = []

        if low < 0:
            out.extend(self._slice_low_pad(field=field, index=(low, 0), padding=padding))

        out.extend(self.data[field][(max(low, 0)):(min(self.steps, high))])

        if high > self.steps: # self.steps
            out.extend([self._high_pad(field=field, index=self.steps+1)] * len(range(self.steps, high)))

        return out
    
    def borders(self):
        return self.has_start, self.has_end
    
    def extend(self, other):
        
        assert self.envhash == other.envhash
        
        assert not self.has_end
        
        assert not other.has_start
        
        for field in self.data.keys():
            self.data[field].extend(other.data[field])
            
        self.steps += other.steps
        self.has_end = other.has_end
        self.to_be_extended = other.to_be_extended
        self.terminal_ob = other.terminal_ob
        
        
class PathBranch(Path):
    
    def __init__(self, 
                 trunk_path: Path,
                 fork_transition: int):
        
        super().__init__()
        
        self.trunk_path = trunk_path
        self.fork_transition = fork_transition
        self.is_real = False
        
        # ensure that path will never be extended
        self.has_start = True
        self.has_end = True
        self.to_be_extended = False
        
    def _low_pad(self, field: str, index: int, padding='first'):
        
        assert index < 0
        
        # special case for lagging 'next_obs'
        shift = 0
        if field == 'next_obs':
            field = 'obs'
            shift += self.next_obs_n
        
        out = self.trunk_path.get(field=field, 
                                  index=self.fork_transition+index+shift, 
                                  padding=padding)
        
        return out
    
    def _slice_low_pad(self, field: str, index: tuple, padding='first'):
        
        low, high = index
        
        assert low < 0 and high <= 0
        
        # special case for lagging 'next_obs'
        shift = 0
        if field == 'next_obs':
            field = 'obs'
            shift += self.next_obs_n
        
        out = self.trunk_path.slice_get(field=field, 
                                        index=(self.fork_transition+low+shift, self.fork_transition+high+shift), 
                                        padding=padding)
        
        return out
        

class PathList:
    
    """
    Class for storing environment multiple rollouts,
    implements some useful methods for working with 
    environment rollouts
    """
    
    def __init__(self):
        
        self.rollouts = []
        self.n_transitions = 0
        
        self.next_obs_n = None
        
        self.n_step_rews_n = None
        self.n_step_rews_gamma = None
        
        self.n_gamma_pow_n = None
        self.n_gamma_pow_gamma = None
        
        self.n_step_terminals_n = None
    
    def __add__(self, other):
        out = PathList()
        out.rollouts = self.rollouts + other.rollouts
        out.n_transitions = self.n_transitions + other.n_transitions
        return out
        
    def copy(self):
    	return deepcopy(self)
    
    def n_paths(self):
        return len(self.rollouts)
    
    def mean_path_len(self):
        mean_path_len = None
        if self.n_transitions > 0:
            mean_path_len = self.n_transitions / self.n_paths()
        else:
            warn('No paths in PathList, returning None as mean path length',
                 UserWarning)
        return mean_path_len
    
    def total_rewards_by_path(self):
        rews_sums = np.array([])
        if self.n_transitions > 0:
            rews_sums = np.array(list(
                map(lambda p: p.total_reward(), self.rollouts)
            ))
        else:
            warn('No paths in PathList, returning returning empty np.array',
                 UserWarning)
        return rews_sums
    
    def add_disc_cumsum(self, cumsum_field: str, field: str, gamma: float):
        for path in self.rollouts:
            path.add_disc_cumsum(cumsum_field=cumsum_field,
                                 field=field,
                                 gamma=gamma)
            
    def add_lag_concat(self, lag_field: str, field: str, nlags: int, 
                       concat_axis=-1, expand_axis=None, padding='first'):
        if nlags < 1:
            raise ValueError('Number of lags should be greater than 0')
        for path in self.rollouts:
            path.add_lag_concat(lag_field=lag_field, field=field, nlags=nlags, 
                                concat_axis=concat_axis, expand_axis=expand_axis,
                                padding=padding)
            
    def add_next_obs(self, n=1):
        self.next_obs_n = n 
        for path in self.rollouts:
            path.add_next_obs(n=n)
            
    def add_n_step_returns(self, n: int, gamma: float):
        self.n_step_rews_n = n
        self.n_step_rews_gamma = gamma
        for path in self.rollouts:
            path.add_n_step_returns(n=n, gamma=gamma)
            
    def add_n_gamma_pow(self, n: int, gamma: float):
        self.n_gamma_pow_n = n
        self.n_gamma_pow_gamma = gamma
        for path in self.rollouts:
            path.add_n_gamma_pow(n=n, gamma=gamma)
            
    def add_n_step_terminals(self, n: int):
        self.n_step_terminals_n = n
        for path in self.rollouts:
            path.add_n_step_terminals(n=n)
            
    def get_paths_idxs(self):
        csum = np.cumsum(list(map(lambda path: path.steps, 
                                  self.rollouts))).tolist()
        idxs = list(zip([0] + csum[:-1], csum))
        return idxs
    
    def drop_field(self, field: str):
        for path in self.rollouts:
            path.drop_field(field=field)
        
    def unpack(self, 
               fields=('obs', 'next_obs',
                       'acs', 'rews', 
                       'terminals', 'is_time_limit'),
               squeeze=True):
        
        out_list = []
        if self.n_transitions > 0:
            for field in fields:
                out_list.append(np.concatenate(
                    [rollout.data[field] for rollout in self.rollouts]
                ))
        else:
            warn('Attempting to unpack PathList with no rollouts, returning empty list', 
                 UserWarning)
            
        if len(out_list) == 1 and squeeze:
            out_list = out_list[0]
            
        return out_list
    
    def pack(self, value, field: str):
        idxs = self.get_paths_idxs()
        for (st, en), path in zip(idxs, self.rollouts):
            data_list = value[st:en]
            assert path.steps == len(data_list)
            path.data[field] = data_list
            
    def create_logs(self, prefix=None, extra_metrics=None) -> dict:
        total_rews = self.total_rewards_by_path()
        pstr = ''
        if prefix is not None:
            pstr = f'/{prefix}'
        data_log = {'avg_return'+pstr: total_rews.mean(),
                    'std_return'+pstr: total_rews.std(),
                    'max_return'+pstr: total_rews.max(),
                    'min_return'+pstr: total_rews.min(),
                    'mean_pathlen'+pstr: self.mean_path_len(),
                    'n_paths'+pstr: self.n_paths(),
                    'n_transitions'+pstr: self.n_transitions}
        if extra_metrics is not None:
            data_log = {**data_log, **extra_metrics}
        return data_log
    
    def get_fields_names(self):
        return set(list(chain(*[list(path.data.keys()) for path in self.rollouts])))
    
    def accelerate(self,
                   actor, 
                   model,
                   h: int,
                   tau: float,
                   real_ratio=0,
                   inplace=False,
                   train_sampling=True,
                   cut_tails=False):
        
        if h > 0 and tau > 0:
        
            # preprocess the data
            model_obs, actor_obs, obs, mba_sample = _pre_proc_data(
                data=self,
                actor=actor,
                model=model,
                tau=tau
            )

            # calculating sample data
            lengths = np.array(list(map(lambda path: path.steps, self.rollouts)))
            csum = np.cumsum(np.array([0] + list(lengths[:-1])))
            values = np.arange(0, self.n_paths())
            pathinds = np.repeat(values, lengths)[mba_sample]
            sample_data = np.concatenate([pathinds[:, np.newaxis], 
                                          (mba_sample - csum[pathinds])[:, np.newaxis]], axis=1)

            # generate list of PathBranch's
            path_branches = [PathBranch(trunk_path=self.rollouts[pathind],
                                        fork_transition=trind) for pathind, trind in sample_data]

            # performing synthetic rollouts
            path_branches, transitions_total = _accelerate(
                data=self,
                model_obs=model_obs, 
                actor_obs=actor_obs,
                obs=obs,
                sample_data=sample_data,
                path_branches=path_branches,
                actor=actor, 
                model=model,
                h=h,
                real_ratio=real_ratio,
                train_sampling=train_sampling,
                cut_tails=cut_tails
            )

            del sample_data, mba_sample
            del model_obs, actor_obs

            if inplace:
                # just add to current pathlist
                self.rollouts.extend(path_branches)
                self.n_transitions += transitions_total  
            else:
                out = PathList()
                out.rollouts = path_branches
                out.n_transitions = transitions_total
                return out
            
        else:
            
            if inplace:
                pass
            else:
                return Pathlist() # return an empty pathlist
            
    def decelerate(self):
        
        rollouts = []
        n_transitions = 0
        
        for path in self.rollouts:
            if not isinstance(path, PathBranch):
                rollouts.append(path)
                n_transitions += path.steps
                
        self.rollouts = rollouts
        self.n_transitions = n_transitions
        

class Sampler:
    
    def __init__(self, 
                 env, 
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first'):
        
        # wrapping envirionment and saving actor lag profile
        self.env = env
        # self.actor = actor
        
        # initialising internal states
        self.last_ob = None
        self.obs_buffer = []
        self.samples_made = 0
        
        # actor exploration state
        self.actor_exploration_state = None
        
        # initialising iterational outputs
        self.last_path = Path()
        self.last_path.has_start = True
        self.last_pathlist = PathList()
        
        # Copy actor lag profile
        self.obs_nlags = obs_nlags
        self.obs_concat_axis = obs_concat_axis
        self.obs_expand_axis = obs_expand_axis
        self.obs_padding = obs_padding
        
        self.lags_required = True
        
    def add_last_path_to_pathlist(self):
        
        # Mark the last path with environment hash
        self.last_path.envhash = self.get_env_hash()
        
        # Append last generated path to pathlist
        self.last_pathlist.rollouts.append(self.last_path)
        self.last_pathlist.n_transitions += self.last_path.steps
        
        # Reset last pathlist
        self.last_path = Path()
        
        # mark whether it has a start;
        self.last_path.has_start = self.last_ob is None
        
    def reset(self):    
        # Reset internal states
        self.last_ob = None
        self.obs_buffer = []       
        
    def reset_pathlist(self):
        self.last_pathlist = PathList()
    
    def add_obs(self, obs: np.ndarray):
        
        self.obs_buffer.append(obs)
        
        if self.lags_required and self.obs_nlags > 0:
            if len(self.obs_buffer) < (self.obs_nlags + 1):
                # create pseudo-path for padding
                pp = Path()
                pp.data['obs'].append(obs)
                pp.steps += 1
                # do padding
                self.obs_buffer = [pp.get('obs', -1, self.obs_padding)] * self.obs_nlags + self.obs_buffer
            self.obs_buffer = self.obs_buffer[-(self.obs_nlags + 1):]
        else:
            self.obs_buffer = [self.obs_buffer[-1]]
            
    def get_obs(self):
        
        out_obs = None
        
        if self.lags_required and self.obs_nlags > 0:
            data_list = self.obs_buffer 
            if self.obs_expand_axis is not None:
                data_list = [np.expand_dims(data_list_i, 
                                            axis=self.obs_expand_axis) for data_list_i in data_list]
            out_obs = np.concatenate(data_list, axis=self.obs_concat_axis)
        else:
            out_obs = self.obs_buffer[-1]
        
        return out_obs
    
    def get_samples_made(self):
        return self.samples_made
    
    def get_env_hash(self):
        return hash(self.env)
        
    def get_env_seed(self):
        return self.env.seed()
    
    def reset_env_if_needed(self):
        # Reset the env in the beggining if needed:
        if self.last_ob is None:
            ob = self.env.reset()
        else:
            ob = self.last_ob
        return ob
    
    def reset_actor_expl_state_if_needed(self, actor):
        if actor.train_sampling:
            if hasattr(actor, 'exploration'):
                if actor.exploration is not None:
                    if self.last_ob is None:
                        actor.exploration.reset_state()
                    
    def load_actor_expl_state(self, actor):
        if actor.train_sampling:
            if hasattr(actor, 'exploration'):
                if actor.exploration is not None:
                    actor.exploration.load_state(
                        state=self.actor_exploration_state
                    )
                
    def save_actor_expl_state(self, actor):
        if actor.train_sampling:
            if hasattr(actor, 'exploration'):
                if actor.exploration is not None:
                    self.actor_exploration_state = actor.exploration.save_state()
    
    def add_obs_and_get_it(self, ob: np.ndarray) -> np.ndarray:
        self.add_obs(ob)
        _ob = self.get_obs()
        return _ob
    
    def step_env(self, ac: np.ndarray):
        self.samples_made += 1
        return self.env.step(ac)
    
    def get_pathlist(self):
        out = self.last_pathlist
        self.reset_pathlist()
        return out
    
    def sample(self, actor,
               n_transitions: int, 
               max_path_length=None,
               reset_when_not_done=False,
               train_sampling=False,
               info_time_limit_key='TimeLimit.truncated') -> PathList:
        
        # Change sampling mode
        previous_mode = actor.train_sampling
        actor.train_sampling = train_sampling
        
        # initializing outer loop:
        transitions_sampled = 0
        self.load_actor_expl_state(actor=actor)
        
        while transitions_sampled < n_transitions:
            
            # updating counters
            transitions_sampled += 1
            
            # get initial observation
            ob = self.reset_env_if_needed()
            self.reset_actor_expl_state_if_needed(actor=actor)

            # update the las path
            self.last_path.data['obs'].append(ob)

            # resolving lags
            _ob = self.add_obs_and_get_it(ob=ob)
            
            # getiing actions from the actor
            _ob = np.expand_dims(_ob, axis=0) # put _ob into batchmode for consistency
            ac = actor.get_action(obs=_ob)
            ac = ac.squeeze(axis=0) # squeeze ac back for storage
            
            # performing steps in the environment
            ob, rew, done, info = self.step_env(ac)

            # adding transitions to path
            self.last_path.data['rews'].append(rew)
            self.last_path.data['acs'].append(ac)
            self.last_path.data['terminals'].append(done) # not rollout_done as before
            
            # handling time limit
            if info_time_limit_key in info.keys():
                tlimit = info[info_time_limit_key]
            else:
                tlimit = False
            self.last_path.data['is_time_limit'].append(tlimit)
            
            self.last_path.steps += 1
            # Mark whether the path is completed in environment
            self.last_path.has_end = done

            # decide whether to finish the path
            if max_path_length is not None:
                rollout_done = done or (self.last_path.steps == max_path_length)
            else:
                rollout_done = done 

            # filling 'terminal_ob' attribute:
            if rollout_done or transitions_sampled >= n_transitions:
                self.last_path.terminal_ob = ob

            # Mark if path will be updated later
            if not rollout_done and not reset_when_not_done:
                self.last_path.to_be_extended = True
            else:
                self.last_path.to_be_extended = False

            # mark what to do after transition is sampled
            if rollout_done:
                self.reset()
                self.add_last_path_to_pathlist()
            else:
                if transitions_sampled >= n_transitions:
                    if reset_when_not_done:
                        self.reset()
                        self.add_last_path_to_pathlist()
                    else:
                        self.last_ob = ob
                        self.add_last_path_to_pathlist()
                else:
                    self.last_ob = ob
        
        self.save_actor_expl_state(actor=actor)
        
        # return previous sampling mode
        actor.train_sampling = previous_mode
        
        return self.get_pathlist()
    
    def sample_n_episodes(self, actor,
                          n_episodes: int,
                          max_path_length=None,
                          train_sampling=False,
                          info_time_limit_key='TimeLimit.truncated') -> PathList:
        
        # Change sampling mode
        previous_mode = actor.train_sampling
        actor.train_sampling = train_sampling
        
        # initializing outer loop:
        self.load_actor_expl_state(actor=actor)
        
        for _ in range(n_episodes):
        
            rollout_done = False
            
            while not rollout_done:
                
                # get initial observation
                ob = self.reset_env_if_needed()
                self.reset_actor_expl_state_if_needed(actor=actor)

                # update the last path
                self.last_path.data['obs'].append(ob)

                # resolving lags
                _ob = self.add_obs_and_get_it(ob=ob)

                # getiing actions from the actor
                _ob = np.expand_dims(_ob, axis=0) # put _ob into batchmode for consistency
                ac = actor.get_action(obs=_ob)
                ac = ac.squeeze(axis=0) # squeeze ac back for storage

                # performing steps in the environment
                ob, rew, done, info = self.step_env(ac)

                # adding transitions to path
                self.last_path.data['rews'].append(rew)
                self.last_path.data['acs'].append(ac)
                self.last_path.data['terminals'].append(done) # not rollout_done as before
                
                # handling time limit
                if info_time_limit_key in info.keys():
                    tlimit = info[info_time_limit_key]
                else:
                    tlimit = False
                self.last_path.data['is_time_limit'].append(tlimit)
                
                self.last_path.steps += 1
                # Mark whether the path is completed in environment
                self.last_path.has_end = done

                # decide whether to finish the path
                if max_path_length is not None:
                    rollout_done = done or (self.last_path.steps == max_path_length)
                else:
                    rollout_done = done 
                    
                # filling 'terminal_ob' attribute:
                if rollout_done:
                    self.last_path.terminal_ob = ob

                # Mark if path will be updated later
                if not rollout_done:
                    self.last_path.to_be_extended = True
                else:
                    self.last_path.to_be_extended = False

                # mark what to do after transition is sampled
                if rollout_done:
                    self.reset()
                    self.add_last_path_to_pathlist()
                else:
                    self.last_ob = ob
        
        self.save_actor_expl_state(actor=actor)
        
        # return previous sampling mode
        actor.train_sampling = previous_mode
        
        return self.get_pathlist()
            
    
class ParallelSampler:
    
    def __init__(self, 
                 env, 
                 obs_nlags=0,
                 obs_concat_axis=-1,
                 obs_expand_axis=None,
                 obs_padding='first', 
                 gpus_share=0.5):
        
        if not isinstance(env, list):
            raise ValueError(
                f"For {type(self).__name__} env arg should "
                "be a list of differently initialized envs")
            
        if len(env) > mp.cpu_count():
            warn(f"You have provided more envs ({len(env)}) "
                 f"than you have cores ({mp.cpu_count()}). "
                 f"Using only first {mp.cpu_count()} envs",
                 UserWarning)
        
        self.n_workers = min(mp.cpu_count(), len(env))

        if not ray.is_initialized():
            ray.init(num_gpus=torch.cuda.device_count())
            
        self.samplers = []
        self.gpu_per_worker = (torch.cuda.device_count() * gpus_share) / self.n_workers
        RemoteSampler = ray.remote(num_gpus=self.gpu_per_worker, 
                                   num_cpus=1)(Sampler)
        
        for env_i in env[:mp.cpu_count()]:
            self.samplers.append(
                RemoteSampler.remote(env=env_i, 
                                     obs_nlags=obs_nlags,
                                     obs_concat_axis=obs_concat_axis,
                                     obs_expand_axis=obs_expand_axis,
                                     obs_padding=obs_padding)
            )

    def get_samples_made(self):
        return sum(ray.get([smp.get_samples_made.remote() for smp in self.samplers]))
    
    def sample(self, actor,
               n_transitions: int, 
               max_path_length=None,
               reset_when_not_done=False,
               train_sampling=False,
               info_time_limit_key='TimeLimit.truncated') -> PathList:
        
        transitions_per_core = ceil(n_transitions / self.n_workers)
        out = PathList()
        
        pathlists = ray.get([
            smp.sample.remote(actor=actor,
                              n_transitions=transitions_per_core,
                              max_path_length=max_path_length,
                              reset_when_not_done=reset_when_not_done,
                              train_sampling=train_sampling,
                              info_time_limit_key=info_time_limit_key) for smp in self.samplers
        ])
        
        for pathlist in pathlists:
            out += pathlist
            
        return out
    
    def sample_n_episodes(self, actor,
                          n_episodes: int,
                          max_path_length=None,
                          train_sampling=False,
                          info_time_limit_key='TimeLimit.truncated') -> PathList:
        
        episodes_per_core = ceil(n_episodes / self.n_workers)
        out = PathList()
        
        pathlists = ray.get([
            smp.sample_n_episodes.remote(actor=actor,
                                         n_episodes=episodes_per_core,
                                         max_path_length=max_path_length,
                                         train_sampling=train_sampling,
                                         info_time_limit_key=info_time_limit_key) for smp in self.samplers
        ])
        
        for pathlist in pathlists:
            out += pathlist
            
        return out 
