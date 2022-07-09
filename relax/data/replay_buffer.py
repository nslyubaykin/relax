import numpy as np

from copy import deepcopy
from itertools import chain
from datetime import datetime
from collections import OrderedDict

from relax.data.sampling import PathList, Path, PathBranch
from relax.data.utils import concat_expand_lags
from relax.data.acceleration import _pre_proc_data, _accelerate
from relax.data.prioritization import learner_id, sample_unique
from relax.data.prioritization import UniformReference, SumTree

    
class ReplayBuffer(PathList):
    
    """
    Class for Replay Buffer, that consists of
    ordered Path objects obtained during the
    sampling from the envirionment
    """
    
    def __init__(self, 
                 size, 
                 fields=('obs',
                         'acs', 
                         'rews', 
                         'terminals',
                         'is_time_limit'),
                 prioritized_learners=[],
                 init_priority=1,
                 init_alpha=1,
                 recalc_tree_every=int(3e6)):
        
        super(ReplayBuffer, self).__init__()
        
        self.size = size
        self.fields = fields
        self.n_additions = 0
        self.transitions_seen = 0
        self.lengths = []
        self.n_paths_added = 0
        self.n_paths_evicted = 0
        
        # Initialising base prioritizer 
        self.prioritizers = {'uniform': UniformReference(size=self.size)}
        
        # Initializing aditional prioritizers if needed
        for pl in prioritized_learners:
            if hasattr(pl, 'prioritized_sampling'):
                if pl.prioritized_sampling:
                    self.prioritizers[learner_id(pl)] = SumTree(
                        size=self.size,
                        init_max=init_priority,
                        init_alpha=init_alpha,
                        recalc_tree_every=recalc_tree_every
                    )
                else:
                    warn(f"Skipped for {type(pl).__name__}: "
                         "attribute 'prioritized_sampling' set "
                         "to 'False'. " 
                         "Set it to 'True' in .__init__() "
                         "in order to use prioritization")
            else:
                warn(f"Skipped for {type(pl).__name__}: "
                     "prioritization is not supported")
        
    def __add__(self, other):
        raise NotImplementedError(f'Addition is not defined for {type(self).__name__}')
        
    def accelerate(self,
                   actor, 
                   model,
                   h: int,
                   tau: float,
                   real_ratio=0,
                   inplace=False,
                   train_sampling=True,
                   cut_tails=False):
        raise NotImplementedError("Direct acceleration is not defined "
                                  f"for {type(self).__name__}, only for its sample")
        
    def decelerate(self):
        raise NotImplementedError("Direct deceleration is not defined "
                                  f"for {type(self).__name__}, only for its sample")
        
    def add_paths(self, paths: PathList):
        
        # check input type
        assert isinstance(paths, PathList)
        
        # Combine rollouts
        other = paths 
        
        # check if there are any incomplete paths to be completed in recent sample
        paths_no_start_data = {}
        paths_with_start, paths_with_start_lengths = [], []
        paths_with_start_ref = []
        pws_i = 0
        for p in other.rollouts:
            if not p.has_start:
                # {path.envhash: (path, path.steps)}
                paths_no_start_data[p.envhash] = (p, p.steps) 
            else:
                paths_with_start.append(p)
                paths_with_start_lengths.append(p.steps)

                # For prioritising
                if len(self.prioritizers) > 0:
                    p_ref = list(zip([pws_i + self.n_paths_added] * p.steps, list(range(p.steps))))
                    paths_with_start_ref.append(p_ref)
                    pws_i += 1

        # searching for paths with no end in buffer
        paths_no_end, paths_searched = 0, 0
        paths_no_end_data = {}
        while paths_no_end < len(paths_no_start_data.keys()) and paths_searched <= self.n_paths():
            paths_searched += 1
            if not self.rollouts[-paths_searched].has_end and self.rollouts[-paths_searched].to_be_extended:
                paths_no_end += 1
                paths_no_end_data[self.rollouts[-paths_searched].envhash] = paths_searched

        # check for consistency
        assert paths_no_end == len(paths_no_start_data.keys())

        # extend paths with no end
        if paths_no_end > 0:
            for envhash, inv_pathind in paths_no_end_data.items():
                pns, pns_length = paths_no_start_data[envhash]

                # Prioritization section
                if len(self.prioritizers) > 0:
                    p_steps = self.rollouts[-inv_pathind].steps
                    p_ind = self.n_paths_added - inv_pathind
                    pns_ref = list(zip([p_ind] * pns_length, list(range(p_steps, (p_steps+pns_length)))))
                    for sampling_tree in self.prioritizers.values():
                        sampling_tree.buffer_reference[-inv_pathind].extend(pns_ref)

                self.rollouts[-inv_pathind].extend(pns)
                self.lengths[-inv_pathind] += pns_length
            
        # just add paths with start to rollouts:
        self.rollouts += paths_with_start
        self.lengths += paths_with_start_lengths
        self.n_paths_added += len(paths_with_start)

        # Prioritising section
        if len(self.prioritizers) > 0:
            for sampling_tree in self.prioritizers.values():
                sampling_tree.buffer_reference += deepcopy(paths_with_start_ref)
        
        # update the number of transitions
        self.n_transitions += other.n_transitions
        self.transitions_seen += other.n_transitions

        # update number of additions
        self.n_additions += 1

        # Keep only new paths
        if self.n_transitions > self.size:

            resid_transitions = self.n_transitions
            keep_last = self.n_paths()
            omit_last = 0 
            while resid_transitions > self.size:
                resid_transitions -= self.lengths[-keep_last]
                keep_last -= 1
                self.n_paths_evicted += 1
                omit_last += 1

            self.rollouts = self.rollouts[-keep_last:]
            self.lengths = self.lengths[-keep_last:]
            self.n_transitions = resid_transitions

            # Prioritization section
            if len(self.prioritizers) > 0:
                for sampling_tree in self.prioritizers.values():
                    sampling_tree.evict_last_paths(n_paths=omit_last)
            
        # Pushing transition indexes into a sampling tree if needed
        # Done after evicting unneded paths
        if len(self.prioritizers) > 0:
            for sampling_tree in self.prioritizers.values():
                sampling_tree.add_batch(batch_size=other.n_transitions)
                # some sanity check
                assert sampling_tree.n_transitions == self.n_transitions
    
    def finish_paths(self):
        for path in self.rollouts:
            if path.to_be_extended:
                path.to_be_extended = False
                
    def _uniform_sample(self, batch_size):
        
        buffer_ind = self.prioritizers['uniform'].sample(
            k=batch_size
        )
        
        # declare the output
        out_data = {}
        
        # populate it
        for datakey in self.fields:
            out_data[datakey] = list(map(lambda pt: self.rollouts[pt[0]].get(datakey, pt[1]), buffer_ind))
        out_data['index_info'] = buffer_ind
                
        return BufferSample(out_data, batch_size, self)
        
    def _prioritized_sample(self, batch_size, p_learner):
        
        p_learner_key = learner_id(p_learner)
        
        if p_learner_key not in self.prioritizers.keys():
            raise ValueError(
                f"{type(p_learner).__name__} not found in buffer prioritization. "
                "Pass it as an elment of list to 'prioritized_learners' argument in " 
                f"{type(self).__name__} .__init__(). "
                "Note: several learners may have independent prioritization."
            )
        
        tree_ind, buffer_ind, p_alpha = self.prioritizers[p_learner_key].sample(
            k=batch_size,
        )
        
        # declare the output
        out_data = {}
        
        # populate it
        for datakey in self.fields:
            out_data[datakey] = list(map(lambda pt: self.rollouts[pt[0]].get(datakey, pt[1]), buffer_ind))
        out_data['index_info'] = buffer_ind
        out_data['tree_index'] = tree_ind
        out_data['p_alpha'] = p_alpha
                
        return BufferSample(out_data, batch_size, self)
    
    def sample(self, batch_size, p_learner=None):
        
        if p_learner is None:
            out_sample = self._uniform_sample(batch_size=batch_size)
        else:
            out_sample = self._prioritized_sample(batch_size=batch_size,
                                                  p_learner=p_learner)
            
        return out_sample
    
    
class BufferSample(object):
    
    def __init__(self, data, n_transitions, parent_buffer):
        
        self.data = data
        self.synth_rollouts = None
        self.n_transitions = n_transitions
        self.pb_n_add = parent_buffer.n_additions
        self.parent_buffer = parent_buffer
        
        self.pathlengths = None
        
        self.next_obs_n = None
        
        self.n_step_rews_n = None
        self.n_step_rews_gamma = None
        
        self.n_gamma_pow_n = None
        self.n_gamma_pow_gamma = None
        
        self.n_step_terminals_n = None
    
    def unpack(self, 
               fields=('obs', 'acs', 'rews', 'terminals', 'is_time_limit'),
               squeeze=True):
        
        out_list = []
        if self.n_transitions > 0:
            for field in fields:
                out_list.append(np.concatenate(
                    [self.data[field]]
                ))
        else:
            warn(f'Attempting to unpack {type(self).__name__} with no transitions, returning empty list', 
                 UserWarning)
            
        # unpacking synthetic transitions if any
        if self.synth_rollouts is not None:
            
            # exclude 'index_info'
            synth_fields = [f for f in fields if f != 'index_info']
            
            if len(synth_fields) > 0: 
                
                # arrange into dict
                out_dict = OrderedDict((f, [o]) for f, o in zip(fields, out_list))
                
                # retreive required data
                synth_out_list = self.synth_rollouts.unpack(fields=synth_fields,
                                                            squeeze=False)
                
                # append to original data:
                for sfield, sval in zip(synth_fields, synth_out_list):
                    out_dict[sfield].append(sval)
                
                out_list = [np.concatenate(out_dict[field], axis=0) for field in fields]
            
        if len(out_list) == 1 and squeeze:
            out_list = out_list[0]
            
        return out_list
    
    def update_priorities(self, p_learner, p, alpha):
        
        p_learner_key = learner_id(p_learner)
        
        if p_learner_key not in self.parent_buffer.prioritizers.keys():
            raise ValueError(
                f"{type(p_learner).__name__} not found in parent buffer prioritization. "
                "Pass it as an elment of list to 'prioritized_learners' argument in " 
                f"{type(self.parent_buffer).__name__} .__init__(). "
                "Note: several learners may have independent prioritization."
            )
        
        self.parent_buffer.prioritizers[p_learner_key].update_priorities(
            tree_ind=self.data['tree_index'],
            p=p, 
            alpha=alpha
        )
        
    def get_priority_sum(self, p_learner):
        
        p_learner_key = learner_id(p_learner)
        
        if p_learner_key not in self.parent_buffer.prioritizers.keys():
            raise ValueError(
                f"{type(p_learner).__name__} not found in parent buffer prioritization. "
                "Pass it as an elment of list to 'prioritized_learners' argument in " 
                f"{type(self.parent_buffer).__name__} .__init__(). "
                "Note: several learners may have independent prioritization."
            )
            
        return self.parent_buffer.prioritizers[p_learner_key].total
    
    def add_lag_concat(self, lag_field: str, field: str, 
                       nlags: int, concat_axis=-1, 
                       expand_axis=None, padding='first'):
        
        # checking if replay buffer has changed since sampling
        if self.pb_n_add != self.parent_buffer.n_additions:
            raise AssertionError('Parent buffer has changed since sampling, cannot query lags')
                 
        # adding lags to synthetic transitions if any
        if self.synth_rollouts is not None:
            self.synth_rollouts.add_lag_concat(
                lag_field=lag_field, 
                field=field, 
                nlags=nlags, 
                concat_axis=concat_axis, 
                expand_axis=expand_axis, 
                padding=padding
            )

        lags = []
        index_info = self.unpack(['index_info'])
        pathindxs, trindxs = index_info[:, 0], index_info[:, 1]

        # special case for lagging 'next_obs'
        shift = 0
        if field == 'next_obs':
            
            field = 'obs'
            shift += self.next_obs_n
            
            # pad correctly the lagged obs
            if self.next_obs_n > 1:
                
                if self.pathlengths is None:
                    pathlengths = np.array(
                        [self.parent_buffer.rollouts[pind].steps for pind in pathindxs]
                    )
                    self.pathlengths = pathlengths
                else:
                    pathlengths = self.pathlengths
                
                truncate = (trindxs + self.next_obs_n) > pathlengths
                trindxs[truncate] = pathlengths[truncate] - self.next_obs_n
                
                del pathlengths, truncate

        # query lags in a slice form
        lag_data_list = [self.parent_buffer.rollouts[pathindx].slice_get(field=field, 
                                                                         index=(trindx-nlags, trindx+1), 
                                                                         padding=padding) \
                         for pathindx, trindx in zip(pathindxs, trindxs + shift)]

        lags = concat_expand_lags(lags_list=lag_data_list, 
                                  concat_axis=concat_axis, 
                                  expand_axis=expand_axis)

        self.data[lag_field] = lags
        
        del lag_data_list, lags, index_info, pathindxs, trindxs
        
    def add_next_obs(self, n=1):
        
        # checking if replay buffer has changed since sampling
        if self.pb_n_add != self.parent_buffer.n_additions:
            raise AssertionError('Parent buffer has changed since sampling, cannot query next_obs')
            
        self.next_obs_n = n 
         
        next_obs = []
        index_info = self.unpack(['index_info'])
        pathindxs, trindxs = index_info[:, 0], index_info[:, 1]
        
        if self.n_transitions > 0:
            
            for pathindx, trindx in zip(pathindxs, trindxs + self.next_obs_n):
                next_obs.append(
                    self.parent_buffer.rollouts[pathindx].get('obs', trindx)
                )
            
        else:
            warn('Batch has not been sampled, returning empty list',
                 UserWarning)
            
        self.data['next_obs'] = next_obs
        
        # adding next_obs to synthetic transitions if any
        if self.synth_rollouts is not None:
            self.synth_rollouts.add_next_obs(n=n)
            
        del index_info, pathindxs, trindxs, next_obs
            
    def add_n_step_terminals(self, n: int):
        
        # checking if replay buffer has changed since sampling
        if self.pb_n_add != self.parent_buffer.n_additions:
            raise AssertionError('Parent buffer has changed since sampling, cannot query terminals')
        
        self.n_step_terminals_n = n

        index_info = self.unpack(['index_info'])

        terminals = [self.parent_buffer.rollouts[pathind].get(
            field='terminals',
            index=trind+self.n_step_terminals_n-1
        ) for pathind, trind in index_info]
        
        self.data['n_step_terminals'] = terminals
        
        # adding n_step_terminals to synthetic transitions if any
        if self.synth_rollouts is not None:
            self.synth_rollouts.add_n_step_terminals(n=n)
            
        del index_info, terminals
            
    def add_n_step_returns(self, n: int, gamma: float):
        
        # checking if replay buffer has changed since sampling
        if self.pb_n_add != self.parent_buffer.n_additions:
            raise AssertionError('Parent buffer has changed since sampling, cannot query rewards')
            
        self.n_step_rews_n = n
        self.n_step_rews_gamma = gamma
        
        index_info = self.unpack(['index_info'])

        gammas = self.n_step_rews_gamma**np.arange(self.n_step_rews_n)

        n_step_rews = np.array(
            [self.parent_buffer.rollouts[pathind].slice_get(
                field='rews',
                index=(trind, trind+self.n_step_rews_n)
            ) for pathind, trind in index_info]
        )

        n_step_rews = (n_step_rews * gammas).sum(axis=-1).tolist()
        
        self.data['n_step_rews'] = n_step_rews
        
        # adding n_step_rews to synthetic transitions if any
        if self.synth_rollouts is not None:
            self.synth_rollouts.add_n_step_returns(n=n, gamma=gamma)
            
        del index_info, gammas, n_step_rews
            
    def add_n_gamma_pow(self, n: int, gamma: float):
        
        # checking if replay buffer has changed since sampling
        if self.pb_n_add != self.parent_buffer.n_additions:
            raise AssertionError('Parent buffer has changed since sampling')
            
        self.n_gamma_pow_n = n
        self.n_gamma_pow_gamma = gamma
        
        index_info = self.unpack(['index_info'])
        pathindxs, trindxs = index_info[:, 0], index_info[:, 1]

        if self.pathlengths is None:
            pathlengths = np.array(
                [self.parent_buffer.rollouts[pind].steps for pind in pathindxs]
            )
            self.pathlengths = pathlengths
        else:
            pathlengths = self.pathlengths

        n_gamma_pow = (self.n_gamma_pow_gamma**np.minimum(self.n_gamma_pow_n, pathlengths - trindxs)).tolist()
        
        self.data['n_gamma_pow'] = n_gamma_pow
        
        # adding n_gamma_pow to synthetic transitions if any
        if self.synth_rollouts is not None:
            self.synth_rollouts.add_n_gamma_pow(n=n, gamma=gamma)
            
        del index_info, pathindxs, trindxs, n_gamma_pow, pathlengths

    def drop_field(self, field: str):
        
        if field in self.data.keys():
            del self.data[field]
            
        # drop field from synthetic transitions if any
        if self.synth_rollouts is not None:
            self.synth_rollouts.drop_field(field=field)
            
    def get_fields_names(self):
        
        real_names = set(self.data.keys())
        
        # get fields nemes from synthetic transitions if any
        if self.synth_rollouts is not None:
            synth_names = self.synth_rollouts.get_fields_names()
            real_names = real_names.union(synth_names)
        
        return real_names
    
    def accelerate(self,
                   actor, 
                   model,
                   h: int,
                   tau: float,
                   real_ratio=0,
                   inplace=False,
                   train_sampling=True,
                   cut_tails=False):
        
        # if already accelerated - decelerate first
        # no multiple degree acceleration like for PathList right now
        if self.synth_rollouts is not None:
            self.decelerate() # later may be added recursive acceleration
        
        if h > 0 and tau > 0:
        
            # preprocess the data
            model_obs, actor_obs, obs, mba_sample = _pre_proc_data(
                data=self,
                actor=actor,
                model=model,
                tau=tau
            )

            # calculating sample data
            sample_data = self.unpack(['index_info'])
            sample_data = sample_data[mba_sample, :]

            # generate list of PathBranch's
            path_branches = [PathBranch(trunk_path=self.parent_buffer.rollouts[pathind],
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
            
            out = PathList()
            out.rollouts = path_branches
            out.n_transitions = transitions_total
            
            del sample_data, mba_sample
            del model_obs, actor_obs

            if inplace:
                # reassign new values
                self.synth_rollouts = out
                self.n_transitions += self.synth_rollouts.n_transitions
            else:     
                return out
            
        else:
            
            if inplace:
                pass
            else:
                return Pathlist() # return an empty pathlist
            
    def decelerate(self):       
        self.n_transitions -= self.synth_rollouts.n_transitions
        del self.synth_rollouts
        self.synth_rollouts = None
        