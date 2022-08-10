# ReLAx
ReLAx - Reinforcement Learning Applications

ReLAx is an object oriented library for deep reinforcement learning built on top of PyTorch.

# Contents 
<!-- toc -->

- [Implemented Algorithms](#implemented-algorithms)
- [Special Features](#special-features)
- [Minimal Examples](#minimal-examples)
  - [On Policy](#on-policy)
  - [Off policy](#off-policy)
- [Usage With Custom Environments](#usage-with-custom-environments)
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
  - Filtering Reward Weigthed Refinement (PDDM): [example](https://github.com/nslyubaykin/relax_frwr_example/blob/master/frwr_tutorial.ipynb)
- Hybrid MB-MF 
  - MBPO: [example](https://github.com/nslyubaykin/relax_mbpo_example/blob/master/mbpo_tutorial.ipynb)
  - DYNA-Q: [example](https://github.com/nslyubaykin/relax_dyna_q_example/blob/master/dyna_q_tutorial.ipynb)

## Special Features
__ReLAx offers a set of special features:__

  - Simple interface for lagging environment observations: [Handling Partial Observability with lagged LSTM Policy](https://github.com/nslyubaykin/rnns_for_pomdp/blob/master/lstm_for_pomdp.ipynb)
  - Sampling from parallel envirionments: [Speeding Up PPO with Parallel Sampling](https://github.com/nslyubaykin/parallel_ppo/blob/master/parallel_ppo.ipynb)
  - Wide possibilities for scheduling hyperparameters: [Scheduling TRPO's KL Divergence Constraint](https://github.com/nslyubaykin/trpo_schedule_kl/blob/master/trpo_schedule_kl.ipynb)
  - Support of N-step bootstrapping for all off-policy value-based algorithms: [Multistep TD3 for Locomotion]()
  - Support of Prioritised Experience Replay for all off-policy value-based algorithms: [Prioritised DQN for \*Env-Name\*]()
  - Simple interface for model-based acceleration: [DYNA Model-Based Axceleration with TD3](https://github.com/nslyubaykin/relax_dyna_q_example/blob/master/dyna_q_tutorial.ipynb)

__And other options for building non-standard RL architectures:__

  - Training PPO with DQN as a critic
  - Model-based accelerated RAINBOW
  - Model-based initialization for SAC
  - [Multi-tasking with model-based RL](https://github.com/nslyubaykin/mbrl_multitasking/blob/master/mbrl_multitasking.ipynb)

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

## Usage With Custom Environments
Some examples how to use ReLAx with user defined envirionments:

  - Playing 2048 with RAINBOW

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

```.bash
```

### Atari Environments

ReLAx package was developed and tested with gym\[atari\]==0.17.2. Newer versions also should work, however, its compatibility with provided Atari wrappers is uncertain.

Here is Gym Atari installation guide:

```.bash
pip install gym[atari]==0.17.2
```
In case of "ROMs not found" error do the following steps:

1) Download ROMs archieve
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
 - Offline RL (CQL, BEAR, BCQ)
 - Decision Transformers
 - PPG
 - QR-DQN
 - IQN
 - Discrete SAC
 - NAF
 - Stochastic envirionment models
 - Improving documentation

## Known Issues
  
  - Lack of documentation (right now compensated with usage examples)
  - Filtering Reward Weigthed Refinement declared performance in paper is not yet reached
  - DYNA-Q is not campatible with PER as it is not clear which priority to assign to synthetic branched transitions (possible option: same priority as its parent transition)
