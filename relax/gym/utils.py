import glob
import io
import base64
import numpy as np

from gym.wrappers import Monitor
from gym import Wrapper
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

from relax.data.sampling import Sampler


def show_video():
    mp4list = glob.glob('content/video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")

def wrap_env(env):
    env = Monitor(env, 'content/video', force=True)
    return env

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)
            
def visualize_actor(actor, env, nsteps=1000,
                    size=(1400, 900), seed=0, 
                    train_sampling=False):
    
    display = Display(visible=0, size=size)
    display.start()

    test_env = wrap_env(env)
    test_env.seed(seed)
    
    obs_nlags = 0
    obs_concat_axis = -1
    obs_expand_axis = None
    obs_padding = 'first'
    
    if hasattr(actor, 'obs_nlags'):
        obs_nlags = actor.obs_nlags
    if hasattr(actor, 'obs_concat_axis'):
        obs_concat_axis = actor.obs_concat_axis
    if hasattr(actor, 'obs_expand_axis'):
        obs_expand_axis = actor.obs_expand_axis
    if hasattr(actor, 'obs_padding'):
        obs_padding = actor.obs_padding
    
    smp = Sampler(env=test_env, 
                  obs_nlags=obs_nlags,
                  obs_concat_axis=obs_concat_axis,
                  obs_expand_axis=obs_expand_axis,
                  obs_padding=obs_padding)
    
    previous_mode = actor.train_sampling
    actor.train_sampling = train_sampling
    ob = test_env.reset()
    tot_rew = 0
    for i in range(nsteps):
        test_env.render(mode='rgb_array')
        smp.add_obs(ob)
        _ob = smp.get_obs()
        _ob = np.expand_dims(_ob, axis=0) # put _ob into batchmode for consistency
        ac = actor.get_action(_ob) 
        ac = ac.squeeze(axis=0) # squeeze ac back for storage
        ob, rew, done, info = smp.step_env(ac)
        tot_rew += rew
        if done:
            break
    test_env.close()
    actor.train_sampling = previous_mode
    print(f'Simulation done, total reward: {tot_rew}')
    print('Loading video...')
    show_video()
