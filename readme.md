# CCPPO and QMIX Reinforcement Learning on Google Research Football

## Overview

This project implements two state-of-the-art Reinforcement Learning (RL) algorithms, **Centralized Critic Proximal Policy Optimization (CCPPO)** and **QMIX**, within the **Google Research Football 3v3** environment. The objective is to train agents that can outperform baseline models by effectively coordinating in a multi-agent setting to maximize scoring opportunities while minimizing conceding goals.

## Features

- **Multi-Agent RL Algorithms**: Implementation of CCPPO and QMIX tailored for the 3v3 football environment.
- **Custom Reward Shaping**: Designed to encourage strategic behaviors such as efficient ball possession and passing.
- **Coordination Metrics**: Metrics developed to quantify and analyze agent cooperation.
- **Hyperparameter Optimization**: Utilizes Population Based Training (PBT) for efficient tuning.
- **Performance Evaluation**: Agents are evaluated against baseline models to assess improvements.

## Environment

### 3v3 Google Research Football

- **Team Composition**: 2 field players and 1 goalkeeper per team.
- **State Representation**: 43-dimensional vector including player positions, ball status, and game mode.
- **Action Space**: 19 discrete actions covering movement, ball handling, and strategies.
- **Termination Conditions**: Goal scored, ball out of bounds, or 500 timesteps.

### Reward Structure

- **Goal Scored**: +1 reward.
- **Advancing Towards Goal**: +1 reward per player for crossing distance checkpoints.
- **Time Penalty**: -0.01 per timestep.
- **Ball Possession**: +0.01 for left team possession, -0.01 for right team.
- **Possession Change**: ±0.05 based on possession shifts.
- **Successful Pass**: +0.1 for forward passes advancing >10m.
- **Shot on Goal**: ±0.1 based on shot effectiveness near the opponent's goal.

## Algorithms

### Centralized Critic Proximal Policy Optimization (CCPPO)

Enhances PPO by incorporating a centralized critic that leverages global state information during training, addressing challenges like non-stationarity and credit assignment in multi-agent environments.

### QMIX

Designed for environments with partial observations, QMIX employs a mixing network to combine individual agent value functions into a joint action-value function, ensuring decentralized execution while maintaining coordinated training.

## Coordination Metrics

1. **Spread**: Percentage of timesteps where teammates are spread >20m apart.
2. **Passing Proportions**: Frequency of short vs. long passes.
3. **Same Sticky Actions**: Proportion of timesteps where agents perform the same action.

## Implementation

- **Training**: Utilizes Ray's RLLib library with PBT for hyperparameter optimization.
- **Evaluation**: Trained agents are evaluated against three baseline agents over 100 episodes.

## Results

- **Win Rate**: CCPPO outperformed all baseline agents, while QMIX beat one baseline.
- **Coordination**: Metrics indicated improved strategic positioning, passing behavior, and reduced action redundancy for CCPPO.

## Challenges

- **Compute Constraints**: Limited resources hindered extensive hyperparameter tuning and experimentation.
- **Sample Efficiency**: PPO's need for numerous episodes made training time-consuming.
- **Local Optima**: Agents often plateaued around a 50% win rate.
- **Reward Design**: Balancing rewards to prevent exploitation of short-term gains over long-term success.

## Scripts

- **`football_tools.py`**: Defines custom callbacks for coordination metrics and custom reward shaping.
- **`train_agent_cc.py`**: Trains a CCPPO agent using the custom rewards.
- **`train_group_agents.py`**: Trains agents using the QMIX algorithm with PBT.
- **`centralized_critic_models.py` & `centralized_critic_2.py`**: Modified from Ray RLLib to fit the football environment, used in `train_agent_cc.py`.
