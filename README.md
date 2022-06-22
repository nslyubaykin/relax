# ReLAx
ReLAx - Reinforcement Learning Applications

ReLAx is an object oriented library for deep reinforcement learning built on top of PyTotch.

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
    - VPG
    - TRPO
    - PPO
  - Off-policy
    - DQN 
    - Double DQN
    - Dueling DQN
    - Noisy DQN
    - Categorical DQN
    - RAINBOW
    - DDPG
    - TD3
    - SAC
- Model Based:
  - Random Shooting
  - Cross Entropy Method (CEM)
  - Filtering Reward Weigthed Refinement (PDDM)
- Hybrid MB-MF 
  - MBPO
  - DYNA-Q

## Special Features
ReLAx offers a set of special features:

  - Simple interface for lagging environment observations: Handling Patial Observability with lagged LSTM Policy
  - Sampling from parallel envirionments: Speeding Up PPO with Parallel Sampling
  - Wide possibilities for scheduling hyperparameters: Scheduling TRPO's KL Divergence Constraint
  - Support of N-step bootstrapping for all off-policy value-based algorithms: Multistep TD3 for Locomotion
  - Support of Prioritised Experience Replay for all off-policy value-based algorithms: Prioritised DQN for *Env-Name*
  - Simple interface for model-based axceleration: DYNA Model-Based Axceleration with TD3

And other options for building non-standard RL architectures:

  - Training PPO with DQN as a critic
  - Model-based axcelerated RAINBOW
  - Model-based initialization for SAC

## Minimal Examples

### On Policy

```python
# PPO 
pass
```

### Off policy

```python
# DQN
pass
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
source activate relax
pip install -r requirements.txt
pip install -e .
```

### Mujoco

```.bash
```

### Atari Environments

```.bash
```

## Further Developments
In the future the following functionality is planned to be added:

 - Curiosity (RND)
 - Offline RL (BEAR, BCQ)
 - Decision Transformers
 - QR-DQN
 - IQN
 - Discrete SAC
 - NAF
 - Stochastic envirionment models
 - Improving documentation

## Known Issues

  - Lack of documentation
  - Filtering Reward Weigthed Refinement declared performance in paper is not yet reached
  - DYNA-Q is not campatible with PER as it is not clear which priority to assign to synthetic branched transitions
