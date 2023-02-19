import gym
import torch
import pytest

import numpy as np

from relax.rl.actors import Random
from relax.data.sampling import Path, PathList, Sampler


EPS = 1e-8


@pytest.fixture
def lander_env():
    env = gym.make('LunarLander-v2')
    env.seed(42)
    return env

@pytest.fixture
def lander_sampler(lander_env):
    return Sampler(env=lander_env)

@pytest.fixture
def lander_random_actor(lander_env):
    return Random(env=lander_env)

@pytest.fixture
def lander_pathlist(lander_sampler, lander_random_actor):
    pathlist =  lander_sampler.sample(
        actor=lander_random_actor,
        n_transitions=5000,
        max_path_length=None, 
        reset_when_not_done=False,
        train_sampling=False
    )
    return pathlist

def test_disc_cumsum(lander_pathlist):
    
    gamma = 0.99
    
    # select first path
    path = lander_pathlist.rollouts[0]
    
    # manually calculate reward-to-go
    rews = np.array(path.data['rews'])
    gammas = np.array([gamma**i for i in range(len(rews))])
    rtg_manual = (rews * gammas).sum()
    
    # calculate reward-to-go using Path's method
    path.add_disc_cumsum(
        cumsum_field='rtg', field='rews', gamma=gamma
    )
    
    # check if they are equal
    assert np.abs(path.data['rtg'][0] - rtg_manual) < EPS
