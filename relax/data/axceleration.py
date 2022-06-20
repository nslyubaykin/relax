import numpy as np

from relax.schedules import init_schedule
from relax.data.utils import handle_lags, get_last_lag, get_next_lag_obs


def _pre_proc_data(data,
                   actor,
                   model,
                   tau: float) -> tuple:
    
    # do not recalculate if lag profile is the same
    # create config string and add to lagged variable name
    actor_config = '_'.join([str(l) for l in [actor.obs_nlags, # what if actor imply no lags???
                                              actor.obs_concat_axis, 
                                              actor.obs_expand_axis, 
                                              actor.obs_padding]])

    model_config = '_'.join([str(l) for l in [model.obs_nlags, 
                                              model.obs_concat_axis, 
                                              model.obs_expand_axis, 
                                              model.obs_padding]])

    # Preparing the data for actor
    # creating lags if needed in model
    # unpack rollouts for training
    actor_obs = handle_lags(data=data,
                            fields={'obs':'lag_concat_obs_' + actor_config,},
                            nlags=actor.obs_nlags,
                            concat_axis=actor.obs_concat_axis,
                            expand_axis=actor.obs_expand_axis,
                            padding=actor.obs_padding)

    # Preparing the data for model
    # creating lags if needed in model
    # unpack rollouts for training
    model_obs = handle_lags(data=data,
                            fields={'obs':'lag_concat_obs_' + model_config,},
                            nlags=model.obs_nlags,
                            concat_axis=model.obs_concat_axis,
                            expand_axis=model.obs_expand_axis,
                            padding=model.obs_padding)

    # drop these temporary created fields:
    data.drop_field(field='lag_concat_obs_' + actor_config)
    data.drop_field(field='lag_concat_obs_' + model_config)
    
    # Sample random transitions according to tau
    int_part = tau // 1
    frac_part = tau % 1
    mba_sample = []
    
    if int_part > 0:
        mba_sample.append(
            np.repeat(np.arange(data.n_transitions), int_part)
        )
        
    if frac_part > 0:
        sample_size = int(frac_part * data.n_transitions)
        mba_sample.append(
            np.random.choice(data.n_transitions, sample_size, replace=False)
        )
        
    mba_sample = np.concatenate(mba_sample)
    mba_sample.sort()
    
    # apply sampling
    model_obs, actor_obs = model_obs[mba_sample], actor_obs[mba_sample]
    
    # reconstruct original non-lagged observation
    obs = get_last_lag(lag_obs=model_obs, 
                       nlags=model.obs_nlags,
                       concat_axis=model.obs_concat_axis, 
                       expand_axis=model.obs_expand_axis)
    
    return model_obs, actor_obs, obs, mba_sample


def _axcelerate(data,
                model_obs: np.ndarray, 
                actor_obs: np.ndarray,
                obs: np.ndarray,
                sample_data: np.ndarray,
                path_branches: list,
                actor, 
                model,
                h: int,
                real_ratio=0,
                train_sampling=True,
                cut_tails=False) -> tuple:
    
    # init outputs
    pred_obs, pred_rews, pred_acs, pred_terminals = [], [], [], []

    # Change sampling mode
    previous_mode = actor.train_sampling
    actor.train_sampling = train_sampling
    
    # iteratively predict states and rewards over the sequence:
    for _ in range(h+1): # +1 for terminal obs

        # query actor for actions
        acs = actor.get_action(actor_obs)

        # query model for next_obs and rews
        next_obs, rews, terminals = model.forward_np(lag_obs=model_obs, acs=acs)

        # calculate next lag obs for iterative input
        # model
        model_obs_next = get_next_lag_obs(lag_obs=model_obs, next_obs=next_obs, 
                                          nlags=model.obs_nlags, 
                                          concat_axis=model.obs_concat_axis, 
                                          expand_axis=model.obs_expand_axis)

        # actor
        actor_obs_next = get_next_lag_obs(lag_obs=actor_obs, next_obs=next_obs, 
                                          nlags=actor.obs_nlags, 
                                          concat_axis=actor.obs_concat_axis, 
                                          expand_axis=actor.obs_expand_axis)

        # save current step predictions
        pred_obs.append(obs)
        pred_rews.append(rews)
        pred_acs.append(acs)
        pred_terminals.append(terminals)

        # update values for the next step
        model_obs = model_obs_next
        actor_obs = actor_obs_next
        obs = next_obs

    # return previous sampling mode
    actor.train_sampling = previous_mode

    # Formulate resulting tensors
    pred_obs, pred_rews = np.array(pred_obs), np.array(pred_rews)
    pred_acs, pred_terminals = np.array(pred_acs), np.array(pred_terminals)
    
    # cut beyond terminal forecasts if terminal function is given
    true_h = np.clip((pred_terminals.cumsum(axis=0) <= 1).sum(axis=0), a_max=h, a_min=None)
    
    # add some proportion of real data paths if needed
    real_paths = np.random.binomial(n=1, p=real_ratio, size=len(path_branches))
    
    # Write forecasted data to PathBranch'es
    transitions_total = 0

    for i, path_branch in enumerate(path_branches):

        is_real = real_paths[i]

        if not is_real:

            if cut_tails:
                regularized_length = path_branch.trunk_path.steps - path_branch.fork_transition
                regularized_length = min(h, regularized_length)
            else:
                regularized_length = h

            # account for terminals if needed
            regularized_length = min(regularized_length, true_h[i])

            # subset original data
            obs = pred_obs[:, i, :].copy()
            acs = pred_acs[:, i, :].copy()
            rews = pred_rews[:, i].copy()
            terminals = pred_terminals[:, i].copy()

            # pass it to PathBranch
            path_branch.data['obs'] = list(obs)[:regularized_length]
            path_branch.data['acs'] = list(acs)[:regularized_length]
            path_branch.data['rews'] = list(rews)[:regularized_length]
            path_branch.data['terminals'] = list(terminals)[:regularized_length]
            # always False as there is no time limit for synt data:
            path_branch.data['is_time_limit'] = [False] * regularized_length 
            
            path_branch.steps = regularized_length
            path_branch.terminal_ob = obs[regularized_length].copy()

        else:

            regularized_length = path_branch.trunk_path.steps - path_branch.fork_transition
            regularized_length = min(h, regularized_length)

            for field in ['obs', 'acs', 'rews', 'terminals', 'is_time_limit']:
                path_branch.data[field] = path_branch.trunk_path.slice_get(
                    field=field, 
                    index=(path_branch.fork_transition, path_branch.fork_transition+regularized_length)
                )

            path_branch.is_real = True
            path_branch.steps = regularized_length
            path_branch.terminal_ob = path_branch.trunk_path.get(
                field='obs', 
                index=path_branch.fork_transition+regularized_length
            )

        transitions_total += regularized_length
        
    del pred_obs, pred_rews, pred_acs
    del true_h, real_paths
    
    return path_branches, transitions_total


class DynaAxcelerator:
    
    def set_axceleration(self, 
                         model,
                         actor=None,
                         h=1,
                         tau=1,
                         real_ratio=0,
                         train_sampling=True,
                         cut_tails=False):
        
        # if actor is not specified - use itself for axceleration
        if actor is None:
            actor = self
        
        self.axceleration = {
            'model': model,
            'actor': actor,
            'h': init_schedule(h, discrete=True),
            'tau': init_schedule(tau),
            'real_ratio': init_schedule(real_ratio),
            'inplace': True,
            'train_sampling': train_sampling,
            'cut_tails': cut_tails
        }
    
    @property
    def axceleration_config(self):
        out = {}
        for k, v in self.axceleration.items():
            if k in ['h', 'tau', 'real_ratio']:
                out[k] = v.value(self.global_step)
            else:
                out[k] = v
        return out
