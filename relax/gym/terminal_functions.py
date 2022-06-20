import numpy as np

    
def hopper_v2_terminal_fn(obs):
    
    """
    Credits:
    https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/predict_env.py
    """
    
    height = obs[:, 0]
    angle = obs[:, 1]
    not_done = np.isfinite(obs).all(axis=-1) \
               * np.abs(obs[:, 1:] < 100).all(axis=-1) \
               * (height > .7) \
               * (np.abs(angle) < .2)

    done = ~not_done
    done = done[:, None]
    return done.squeeze(axis=-1)

def walker2d_v2_terminal_fn(obs):
    
    """
    Credits:
    https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/predict_env.py
    """

    height = obs[:, 0]
    angle = obs[:, 1]
    not_done = (height > 0.8) \
               * (height < 2.0) \
               * (angle > -1.0) \
               * (angle < 1.0)
    done = ~not_done
    done = done[:, None]
    return done.squeeze(axis=-1)
