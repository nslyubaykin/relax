import numpy as np


def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def disc_cumsum(rewards, gamma):
    
    rewards_vec = np.array(rewards)[np.newaxis, :]

    gammas_vec = np.array(
        [gamma**t for t in range(len(rewards))]
    )[:, np.newaxis]

    gamma_matrix = np.tril(
        np.concatenate([gammas_vec] * len(rewards), axis=1) / gammas_vec.T, 0
    )

    list_of_discounted_cumsums = (gamma_matrix * rewards_vec.T).sum(axis=0).tolist()
    
    return list_of_discounted_cumsums

def handle_lags(data, fields, nlags,
                concat_axis, expand_axis, padding):
    
    if nlags is not None and nlags > 0:
        for key, value in fields.items():
            if value not in data.get_fields_names():
                data.add_lag_concat(lag_field=value,
                                    field=key,
                                    nlags=nlags,
                                    concat_axis=concat_axis,
                                    expand_axis=expand_axis,
                                    padding=padding)
        out = data.unpack(list(fields.values()))
    else:
        out = data.unpack(list(fields.keys()))
    
    return out

def handle_n_step(data,
                  n: int, 
                  gamma: float) -> list:
    
    data.add_next_obs(n=n)
    
    if n > 1:
        
        data.add_n_step_returns(n=n, gamma=gamma)
        data.add_n_gamma_pow(n=n, gamma=gamma)
        data.add_n_step_terminals(n=n)
        out = data.unpack(['n_step_rews', 'n_gamma_pow', 'n_step_terminals'])
    
    else:
        
        rews, terminals = data.unpack(['rews', 'terminals'])
        gamma_pow = np.ones_like(rews) * gamma
        out = [rews, gamma_pow, terminals]
        
    return out

def add_noise(data, mean_data, noise=0.01, eps=1e-6):
    
    mean_data = mean_data.flatten()
    mean_data[mean_data == 0] = eps
    
    noise_std = np.abs(mean_data * noise)
    flat_noise = np.random.multivariate_normal(mean=np.zeros_like(mean_data), 
                                               cov=np.diag(noise_std**2),
                                               size=data.shape[0])
    
    noise_data = np.reshape(flat_noise, data.shape)
    
    return data + noise_data

def get_last_lag(lag_obs, nlags, concat_axis, expand_axis):
    
    if nlags > 0:
        
        shape_vec = np.arange(len(lag_obs.shape))[1:] # exclude batch axis

        concat_axis = shape_vec[concat_axis]
        lag_obs_split = np.split(lag_obs, 
                                 indices_or_sections=1+nlags, 
                                 axis=concat_axis)
        last_lag = lag_obs_split[-1]

        if expand_axis is not None:
            expand_axis = shape_vec[expand_axis]
            last_lag = last_lag.squeeze(axis=expand_axis)

        return last_lag 
    
    else:
        
        return lag_obs
       
def get_next_lag_obs(lag_obs, next_obs, 
                     nlags, concat_axis, expand_axis):
    
    if nlags > 0:

        shape_vec = np.arange(len(lag_obs.shape))[1:] # exclude batch axis

        concat_axis = shape_vec[concat_axis]

        if expand_axis is not None:
            expand_axis = shape_vec[expand_axis]
            next_obs = np.expand_dims(next_obs, axis=expand_axis)

        lag_obs_split = np.split(lag_obs, 
                                 indices_or_sections=1+nlags, 
                                 axis=concat_axis)
        lag_obs_split.append(next_obs)
        lag_obs_next_split = lag_obs_split[-(1+nlags):]
        
        lag_obs_next = np.concatenate(lag_obs_next_split, axis=concat_axis)
        
        return lag_obs_next
    
    else:
        
        return next_obs

def concat_expand_lags(lags_list, concat_axis, expand_axis):
    
    # concat with minimum effort if possible
    if expand_axis is None:
        return list(map(lambda ll: np.concatenate(ll, axis=concat_axis), lags_list))
    
    sample_shape_len = len(lags_list[0][0].shape) + 1 # add batch axis
    
    # add extra dimension if needed
    if expand_axis is not None:
        sample_shape_len += 1

    shape_vec = np.arange(sample_shape_len)[1:] # exclude batch axis
    concat_axis_new = shape_vec[concat_axis]
    expand_axis_new = expand_axis
    
    if expand_axis_new is not None:
        expand_axis_new = shape_vec[expand_axis_new]
    
    concat_list = []
    
    for lag_data in zip(*lags_list):

        lag_data_new = np.array(lag_data)

        if expand_axis_new is not None:
            lag_data_new = np.expand_dims(lag_data_new, axis=expand_axis_new)

        if concat_axis_new != 1:
            lag_data_new = np.swapaxes(lag_data_new, axis1=concat_axis_new, axis2=1)

        concat_list.append(lag_data_new)
        
    # concatenate resulting array
    lags_data = np.concatenate(concat_list, axis=1)

    if concat_axis_new != 1:
        lags_data = np.swapaxes(lags_data, axis1=1, axis2=concat_axis_new)
        
    out = list(lags_data.copy())
    
    del lags_data, concat_list, lag_data, lag_data_new, shape_vec, sample_shape_len, concat_axis_new, expand_axis_new
        
    return out
