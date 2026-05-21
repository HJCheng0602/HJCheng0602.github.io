---
title: Foundation of Reinforcement learning(V)
date: 2026-05-18 21:19:20
description: "SARSA and Q-learning"
tags: 
  - Reinforcement learning
  - review notes
categories:
    - blog
---


## Introduction
In the previous post, we have introduced the estimate of value function: MC and TD. In this post, we will introduce two important algorithms for estimating the action value function: SARSA and Q-learning.

Looking back at our previous post, now we have known 'What is the best state': estimating the state value function $V^{\pi}(S_t)$, but we still don't know 'What is the best action': $\pi(s) = \arg\max_{a \in A}P(s'|s, a)V^{\pi}(s')$. Here we don't know the transition probability $P(s'|s, a)$, so we can't directly compute the optimal policy. So, we need to estimate the action value function $Q^{\pi}(s, a)$.

## SARSA
For any (state, action, reward, next state, next action) executated by the policy $\pi$, we can update the action value function as follows:
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]
$$