---
title: Foundation of Reinforcement learning(III)
date: 2026-05-17 10:34:13
description: "Model based Reinforcement learning, study for the coming exam"
tags: 
  - Reinforcement learning
  - review notes
categories:
    - blog
---

## Introduction
In the previous post, we have introduced the Bellman equation and the linear programming formulation for MDP. In this post, we will discuss the model-based Reinforcement learning, which is a method to solve the MDP when we do not have the model of the environment.
## The Settings of RL
Typically, RL is framed as MDP, exploring the enviroment and learning the optimal policy.
Generally, we can only observe the episodes and usually, we do not have the model of the environment.

So we need to introduce the model into our RL project. Model-based RL actually is a method to solve the MDP.

## Dynamic Programming Based RL

### Dynamic Programming for finite MDP

Our objective function is simple, just the expected return, which is defined as:
$$
\max_{\pi} E_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} | s_0 = s \right]
$$

In the first episode of the series, we introduced **Backward induction**, which we can start from the last step and then recursively solve the problem. However, this method is not efficient for large state space, and it requires the state transition.

But meanwhile, we can also use the Bellman equation of value function to tackle the problem:
$$
V^{\pi}(s) = \sum_{a \in A} \pi(a|s) 
\left[ 
  \underset{\text{immediate reward}}{\underbrace{R(s,a)}} +
  \underset{\text{discount}}{\underbrace{\gamma}} 
  \sum_{s' \in S} \underset{\text{transition}}{\underbrace{P(s'|s,a)}} 
  \underset{\text{future value}}{\underbrace{V^{\pi}(s')}} 
\right]
$$

### Optimal Value Function
For a state $s$, we can define the optimal value function as the maximum value function over all policies:
$$
V^*(s) = \max_{\pi} V^{\pi}(s)
$$
So the optimal value function is as follows:
$$
V^\ast(s) = \max_{a \in A} R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V^\ast(s')
$$

So the best policy can be derived from the optimal value function as:
$$
\pi^*(a|s) = \arg\max_{a \in A} R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V^\ast(s')
$$

for any state $s$ and policy $\pi$, there is:
$$
V^\ast(s) = V^{\pi^*}(s) \geq V^{\pi}(s)
$$

Obviously, the value function relates to the policy, so we can iterate the optimal value function and the optimal policy until convergence. They are called **Value Iteration** and **Policy Iteration** respectively. 

## Value Iteration
For an MDP which is finite in both state and action space, we can use the value iteration to solve the problem. The value iteration is as follows:
1. Initialize $V(s) = 0$ for all $s \in S$
2. For each state $s \in S$, update the value function as:
$V(s) \leftarrow \max_{a \in A} R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V(s')$
3. Repeat step 2 until convergence.

> NOTE: There isn't any specific order to update the value function, we can update the value function in any order. But the convergence rate may be different.

### Sync & Async Value Iteration

Sync value iteration need to store two copies of value funtion:
1. For any state $s$, we update the value function as:
$V_{new}(s) \leftarrow \max_{a \in A} R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V_{old}(s')$
2. After updating all states, we copy the new value function to the old value function:
$V_{old} \leftarrow V_{new}$


Async value iteration only need to store one copy of value function:
$V(s) \leftarrow \max_{a \in A} R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V(s')$

## Policy Iteration
The assumption of MDP is the same as the value iteration, which is finite in both state and action space. The policy iteration is as follows:
1. Randomly initialize a policy $\pi$ and a value function $V(s) = 0$ for all $s \in S$
2. Repeat the following steps until convergence:
  1. Policy Evaluation: For each state $s \in S$, update the value function as:
  $V(s) \leftarrow \sum_{a \in A} \pi(a|s) \left[ R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V(s') \right]$
  2. Policy Improvement: For each state $s \in S$, update the policy as:
  $\pi(a|s) \leftarrow \arg\max_{a \in A} R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V(s')$

Obviously, the Policy Iteration will be more expensive than the Value Iteration, since it needs to evaluate the policy in each iteration. However, the Policy Iteration can converge faster than the Value Iteration, since it can update the policy in each iteration.

Let's contrast the two methods:
| Method | Value Iteration | Policy Iteration |
| --- | --- | --- |
| Update | Value function | Policy and value function |
| uses| Bellman optimality equation | Bellman expectation equation |

> NOTE:
> 1. Value iteration is a greedy method, we always use the best.
> 2. Update the value function by Bellman equation in Policy Iteration is expensive.
> 3. For smaller space MDP, Policy Iteration is faster than Value Iteration, but for larger space MDP, Value Iteration is faster than Policy Iteration.
> 4. If there isn't any state transition circle, the value iteration is better.

## Bellman operators

In fact, we have introduced the Bellman operators in the previous post, but we haven't discussed it in detail. 

Why Policy Iteration and Value Iteration can converge to the optimal value function? The key is that Bellman operators are contraction mappings.

Bellman operator is the collection of below functions:

1. Bellman expectation operator, usually denoted as $\mathcal{T}^{\pi}$, which is defined as:

   $$\mathcal{T}^{\pi} V(s) = \sum_{a \in A} \pi(a|s) \left[ R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V(s') \right]$$

2. Bellman optimality operator, usually denoted as $\mathcal{T}^{\ast}$ or $\mathcal{T}$, which is defined as:

   $$\mathcal{T} V(s) = \max_{a \in A} R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V(s')$$

They can be used on the state value function and action value function:

- expectation operator is used in the policy iteration, used for computing 
the value function of a given policy, while is the inner loop of the policy iteration.
- optimality operator is used in the value iteration, used for computing the optimal value function, while is the main loop of the value iteration.

Both the Bellman expectation operator and the Bellman optimality operator can be defined on the action value function and the state value function:
1. Bellman expectation operator on V-function:

  $$\begin{aligned}
  V^{\pi}(s) &= E_{\pi} \left[\sum_{t = 0}^{\infty} \gamma^t R_t \mid s_0 = s\right] \\\\
  &= E_{\pi} \left[R(s_0, a_0) + \gamma \sum_{s' \in S} P(s'|s,a) \pi(a|s) V^{\pi}(s')\right] \\\\
  &= \sum_{a \in A} \pi(a|s) \left[ R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V^{\pi}(s') \right] \\\\
  &= (\mathcal{T}^{\pi} V^{\pi})(s)
  \end{aligned}$$

2. Bellman optimality operator on V-function:

  $$\begin{aligned}
  V^{\ast}(s) &= \max_{\pi} E_{\pi} \left[\sum_{t = 0}^{\infty} \gamma^t R_t \mid s_0 = s\right] \\\\
  &= \max_{a \in A} R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) V^{\ast}(s') \\\\
  &= (\mathcal{T} V^{\ast})(s)
  \end{aligned}$$

3. Bellman expectation operator on Q-function:

  $$\begin{aligned}
  Q^{\pi}(s, a) &= E_{\pi} \left[\sum_{t = 0}^{\infty} \gamma^t R_t \mid s_0 = s, a_0 = a\right] \\\\
  &= E_{\pi} \left[R(s_0, a_0) + \gamma \sum_{s' \in S} P(s'|s,a) \pi(a|s) Q^{\pi}(s', a')\right] \\\\
  &= R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) \sum_{a' \in A} \pi(a'|s') Q^{\pi}(s', a') \\\\
  &= (\mathcal{T}^{\pi} Q^{\pi})(s, a)
  \end{aligned}$$

4. Bellman optimality operator on Q-function:

  $$\begin{aligned}
  Q^{\ast}(s, a) &= \max_{\pi} E_{\pi} \left[\sum_{t = 0}^{\infty} \gamma^t R_t \mid s_0 = s, a_0 = a\right] \\\\
  &= R(s, a) + \gamma \sum_{s' \in S} P(s'|s,a) \max_{a' \in A} Q^{\ast}(s', a') \\\\
  &= (\mathcal{T} Q^{\ast})(s, a)
  \end{aligned}$$

Due to the contraction property of the Bellman operators, we can guarantee the convergence of the value iteration and policy iteration to the optimal value function.



## Model RL
In the above sections, our objective environment is a known MDP, all our methods are based on the assumption that we have the model of the environment, which is the transition probability and the reward function. However, in many real-world scenarios, we do not have the model of the environment, so we need to learn the model from the data.

There are two basic thoughts to learn the model of the environment:
1. learn the state transition probability$P(s'|s, a)$:
   
   $$P(s'|s, a) = \frac{N(s, a, s')}{N(s, a)}$$

   where $N(s, a, s')$ is the number of times we have observed the transition from state $s$ to state $s'$ when taking action $a$, and $N(s, a)$ is the number of times we have observed taking action $a$ in state $s$.

2. learn the reward function $R(s, a)$:

    $$R(s, a) = \textbf{average}\left( R_t | s_t = s, a_t = a \right)$$
  
    where $N(s, a)$ is the number of times we have observed taking action $a$ in state $s$, and $R(s, a)$ is the average reward we have observed when taking action $a$ in state $s$.


The simple simulate algorithm is as follows:
1. randomly initialize a policy $\pi$
2. repeat the following steps until convergence:
  1. collect data by executing the policy $\pi$ in the environment, and store the transition data in a replay buffer.
  2. learn the model of the environment from the replay buffer, which includes learning the state transition probability and the reward function.
  3. solve the MDP with the learned model to get the optimal policy $\pi^*$.

Other method to solve this is not learning the MDP, instead we learn the value function directly from the data, which is called model-free RL, we will discuss it in the next post.

## Conclusion
In this post, we have introduced the model-based Reinforcement learning, which is a method to solve the MDP when we do not have the model of the environment. We have discussed the value iteration and policy iteration, which are two basic methods to solve the MDP. We have also introduced the Bellman operators, which are the key to guarantee the convergence of the value iteration and policy iteration. Finally, we have introduced the simple simulate algorithm, which is a method to learn the model of the environment and solve the MDP with the learned model. In the next post, we will discuss the model-free Reinforcement learning, which is a method to learn the value function directly from the data without learning the model of the environment.