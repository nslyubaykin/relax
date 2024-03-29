# ReLAx
ReLAx - Reinforcement Learning Applications

ReLAx is an object oriented library for deep reinforcement learning built on top of PyTorch.

# Contents 
<!-- toc -->

- [Implemented Algorithms](#implemented-algorithms)
- [Special Features](#special-features)
- [Usage With Custom Environments](#usage-with-custom-environments)
- [Minimal Examples](#minimal-examples)
  - [On Policy](#on-policy)
  - [Off policy](#off-policy)
- [Installation](#installation)
  - [Building from GitHub Source](#building-from-github-source)
  - [Mujoco](#mujoco)
  - [Atari Environments](#atari-environments)
- [Further Developments](#further-developments)
- [Known Issues](#known-issues)

<!-- tocstop -->

## Implemented Algorithms
ReLAx library contains implementations of the following algorithms:

- Value Based (Model-Free):
  - On-Policy
    - VPG: [example](https://github.com/nslyubaykin/relax_vpg_example/blob/master/vpg_example.ipynb)
    - A2C: [example](https://github.com/nslyubaykin/relax_a2c_example/blob/master/a2c_example.ipynb)
    - TRPO: [example](https://github.com/nslyubaykin/relax_trpo_example/blob/master/trpo_example.ipynb)
    - PPO: [example](https://github.com/nslyubaykin/relax_ppo_example/blob/master/ppo_example.ipynb)
  - Off-policy
    - DQN: [example](https://github.com/nslyubaykin/relax_dqn_example/blob/master/dqn_tutorial.ipynb)
    - Double DQN: [example](https://github.com/nslyubaykin/relax_double_dqn_example/blob/master/double_dqn_tutorial.ipynb)
    - Dueling DQN: [example](https://github.com/nslyubaykin/relax_dueling_dqn_example/blob/master/dueling_dqn_tutorial.ipynb)
    - Noisy DQN: [example](https://github.com/nslyubaykin/relax_noisy_dqn_example/blob/master/noisy_dqn_tutorial.ipynb)
    - Categorical DQN: [example](https://github.com/nslyubaykin/relax_categorical_dqn_example/blob/master/categorical_dqn_tutorial.ipynb)
    - RAINBOW: [example](https://github.com/nslyubaykin/relax_rainbow_dqn_example/blob/master/rainbow_dqn_tutorial.ipynb)
    - DDPG: [example](https://github.com/nslyubaykin/relax_ddpg_example/blob/master/ddpg_tutorial.ipynb)
    - TD3: [example](https://github.com/nslyubaykin/relax_td3_example/blob/master/td3_tutorial.ipynb)
    - SAC: [example](https://github.com/nslyubaykin/relax_sac_example/blob/master/sac_tutorial.ipynb)
- Model Based:
  - Random Shooting: [example](https://github.com/nslyubaykin/relax_random_shooting_example/blob/master/random_shooting_tutorial.ipynb)
  - Cross Entropy Method (CEM): [example](https://github.com/nslyubaykin/relax_cem_example/blob/master/cem_tutorial.ipynb)
  - Filtering & Reward Weigthed Refinement (PDDM): [example](https://github.com/nslyubaykin/relax_frwr_example/blob/master/frwr_tutorial.ipynb)
- Hybrid MB-MF 
  - MBPO: [example](https://github.com/nslyubaykin/relax_mbpo_example/blob/master/mbpo_tutorial.ipynb)
  - DYNA-Q: [example](https://github.com/nslyubaykin/relax_dyna_q_example/blob/master/dyna_q_tutorial.ipynb)

## Special Features
__ReLAx offers a set of special features:__

  - Simple interface for lagging environment observations: [Recurrent Policies for Handling Partially Observable Environments](https://github.com/nslyubaykin/rnns_for_pomdp)
  - Sampling from parallel environments: [Speeding Up PPO with Parallel Sampling](https://github.com/nslyubaykin/parallel_ppo)
  - Wide possibilities for scheduling hyper-parameters: [Scheduling TRPO's KL Divergence Constraint](https://github.com/nslyubaykin/trpo_schedule_kl)
  - Support of N-step bootstrapping for all off-policy value-based algorithms: [Multistep TD3 for Locomotion](https://github.com/nslyubaykin/nstep_td3)
  - Support of Prioritized Experience Replay for all off-policy value-based algorithms: [Prioritised DDQN for Atari-2600](https://github.com/nslyubaykin/prioritized_ddqn)
  - Simple interface for model-based acceleration: [DYNA Model-Based Acceleration with TD3](https://github.com/nslyubaykin/relax_dyna_q_example) / [MBPO with SAC](https://github.com/nslyubaykin/relax_mbpo_example)

__And other options for building non-standard RL architectures:__

  - [Training PPO with DQN as a critic](https://github.com/nslyubaykin/ppo_with_dqn_critic)
  - [Multi-tasking with model-based RL](https://github.com/nslyubaykin/mbrl_multitasking)
  
## Usage With Custom Environments
Some examples of how to write custom user-defined environments and use them with ReLAx:

  - [Playing 2048 with RAINBOW](https://github.com/nslyubaykin/rainbow_for_2048)

## Minimal Examples

### On Policy

```python
import torch
import gym

from relax.rl.actors import VPG
from relax.zoo.policies import CategoricalMLP
from relax.data.sampling import Sampler

# Create training and eval envs
env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

# Wrap them into Sampler
sampler = Sampler(env)
eval_sampler = Sampler(eval_env)

# Define Vanilla Policy Gradient actor
actor = VPG(
    device=torch.device('cuda'), # torch.device('cpu') if no gpu available
    policy_net=CategoricalMLP(acs_dim=2, obs_dim=4,
                              nlayers=2, nunits=64),
    learning_rate=0.01
)

# Run training loop:
for i in range(100):
    
    # Sample training data
    train_batch = sampler.sample(n_transitions=1000,
                                 actor=actor,
                                 train_sampling=True)
    
    # Update VPG actor
    actor.update(train_batch)
    
    # Collect evaluation episodes
    eval_batch = eval_sampler.sample_n_episodes(n_episodes=5,
                                                actor=actor,
                                                train_sampling=False)
    
    # Print average return per iteration
    print(f"Iter: {i}, eval score: {eval_batch.create_logs()['avg_return']}")
    
```

### Off policy

```python
import torch
import gym

from relax.rl.actors import ArgmaxQValue
from relax.rl.critics import DQN

from relax.exploration import EpsilonGreedy
from relax.schedules import PiecewiseSchedule
from relax.zoo.critics import DiscQMLP

from relax.data.sampling import Sampler
from relax.data.replay_buffer import ReplayBuffer

# Create training and eval envs
env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

# Wrap them into Sampler
sampler = Sampler(env)
eval_sampler = Sampler(eval_env)

# Define schedules
# First 5k no learning - only random sampling
lr_schedule = PiecewiseSchedule({0: 5000}, 5e-5)
eps_schedule = PiecewiseSchedule({1: 5000}, 1e-3)

# Define actor
actor = ArgmaxQValue(
    exploration=EpsilonGreedy(eps=eps_schedule)
)

# Define critic
critic = DQN(
    device=torch.device('cuda'), # torch.device('cpu') if no gpu available
    critic_net=DiscQMLP(obs_dim=4, acs_dim=2, 
                        nlayers=2, nunits=64),
    learning_rate=lr_schedule,
    batch_size=100,
    target_updates_freq=3000
)

# Provide actor with critic
actor.set_critic(critic)

# Run q-iteration training loop:
print_every = 1000
replay_buffer = ReplayBuffer(100000)

for i in range(100000):
    
    # Sample training data (one transition)
    train_batch = sampler.sample(n_transitions=1,
                                 actor=actor,
                                 train_sampling=True)
                                 
    # Add it to buffer                             
    replay_buffer.add_paths(train_batch)
    
    # Update DQN critic
    critic.update(replay_buffer)
    
    # Update ArgmaxQValue actor (only to step schedules)
    actor.update()
    
    if i > 0 and i % print_every == 0:
      # Collect evaluation episodes
      eval_batch = eval_sampler.sample_n_episodes(n_episodes=5,
                                                  actor=actor,
                                                  train_sampling=False)

      # Print average return per iteration
      print(f"Iter: {i}, eval score: " + \
            f"{eval_batch.create_logs()['avg_return']}, " + \
            "buffer score: " + \
            f"{replay_buffer.create_logs()['avg_return']}")
```

## Installation

### Building from GitHub Source

Installing into a separate virtual environment:
```.bash
git clone https://github.com/nslyubaykin/relax
cd relax
conda create -n relax python=3.6
conda activate relax
pip install -r requirements.txt
pip install -e .
```

### Mujoco

To install Mujoco do the following steps:

```.bash
mkdir ~/.mujoco
cd ~/.mujoco
wget http://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
rm mujoco200_linux.zip
wget http://www.roboti.us/file/mjkey.txt
```
Then, add the following line to the bottom of your bashrc: 

```.bash
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin/
```
Finally, install mujoco_py itself:

```.bash
pip install mujoco-py==2.0.2.2
```
__!Note:__ very often installation crushes with error: `error: command 'gcc' failed with exit status 1`.
To debug this run:

```.bash
sudo apt-get install gcc
sudo apt-get install build-essential
```
And then again try to install `mujoco-py==2.0.2.2`

### Atari Environments

ReLAx package was developed and tested with gym\[atari\]==0.17.2. Newer versions also should work, however, its compatibility with provided Atari wrappers is uncertain.

Here is Gym Atari installation guide:

```.bash
pip install gym[atari]==0.17.2
```
In case of "ROMs not found" error do the following steps:

1) Download ROMs archive
```.bash
wget http://www.atarimania.com/roms/Roms.rar
```
2) Unpack it
```.bash
unrar x Roms.rar
```
3) Install atari_py
```.bash
pip install atari_py
```
4) Provide atari_py with ROMS
```.bash
python -m atari_py.import_roms ROMS
```

## Further Developments
In the future the following functionality is planned to be added:

 - Curiosity (RND)
 - Offline RL (CQL, BEAR, BCQ, SAC-N, EDAC)
 - Decision Transformers
 - PPG
 - QR-DQN
 - IQN
 - FQF
 - Discrete SAC
 - NAF
 - Stochastic environment models
 - Improving documentation

## Known Issues
 
  - Lack of documentation (right now compensated with usage examples)
  - On some systems `relax.zoo.layers.NoisyLinear` seems to leak memory. This issue is very unpredictable and yet not fully understood. Sometimes, installing different versions of PyTorch and CUDA may fix it. If the problem persists, as a workaround, consider not using noisy linear layers.
  - Filtering & Reward Weighted Refinement declared performance in paper is not yet reached
  - DYNA-Q is not compatible with PER as it is not clear which priority to assign to synthetic branched transitions (possible option: same priority as its parent transition)
