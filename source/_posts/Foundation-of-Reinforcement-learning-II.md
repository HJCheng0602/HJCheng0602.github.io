---
title: Foundation of Reinforcement learning(II)
date: 2026-05-17 09:15:46
description: "foundation knowledge of Reinforcement learning, study for the coming exam"
tags: 
  - Reinforcement learning
  - review notes
categories:
    - blog
---

## Introduction
Given the former post where we have introduced the MDP and some basic properties, we are now ready to discuss the MDP-based Reinforcement learning. But first, we need to introduce the solution of MDP.

## Bellman Equation
If we have learned the previous post, we will know that there are two types of value function, state value function and action value function. Their mathematical definitions are as follows:
$$
V^{\pi}(s) = E_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} | s_0 = s \right] 
$$

Instinctively, the state value function is the expected return when we start from state $s$ and follow policy $\pi$. Similarly, the action value function is defined as:

$$
Q^{\pi}(s, a) = E_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} | s_0 = s, a_0 = a \right]
$$

Here, it represents the expected return when we start from state $s$, take action $a$, and then follow policy $\pi$ thereafter.

On the other hand, we have a accumulate reward function, which is defined as:
$$
G_t = R_{t} + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
$$
It can be recursively defined as:
$$
G_t = R_{t} + \gamma G_{t+1}
$$

So, we can rewrite the state value function as:
$$
V^{\pi}(s) = \sum_{a} \pi(a|s) \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^{\pi}(s') \right]
$$
Above is the Bellman expectation equation for the state value function. Now, our question is to choose the best policy $\pi$ to maximize the value function. We can define the optimal state value function as:
$$
V^*(s) = \max_{\pi} V^{\pi}(s)
$$

Due to the Principle of Optimality: **Each stage of an optimal policy must be optimal for the remaining stages**, we can derive the Bellman optimality equation for the state value function as:
$$
V^{\ast}(s) = \max_{a} \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^{\ast}(s') \right]
$$

Above is the Bellman optimality equation for the state value function.

## Linear Programming for MDP

We have a key observation that the Bellman optimality equation can be rewritten as a linear programming problem. The linear programming formulation for MDP is as follows:

$$
\begin{aligned}
\text{minimize} \quad & \sum_{s} V(s) \\\\
\text{subject to} \quad & V(s) \geq R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s'), \\\\
&\quad \forall s \in S, a \in A
\end{aligned}
$$

> Proof:
> The first part is to show that the optimal value function $V^{\ast}$ is a feasible solution to the above linear programming problem. We can see that for any state $s$ and action $a$, we have:
> $$
> V^{\ast}(s) = \max_{a'} \left[ R(s, a') + \gamma \sum_{s'} P(s'|s, a') V^{\ast}(s') \right] \\\\ \geq R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^*(s')
> $$
> Thus, $V^{\ast}$ satisfies the constraints of the linear programming problem.
> The second part is to show that any feasible solution satisfying the constraints must be greater than or equal to $V^{\ast}$.
> Given that the optimal policy $\pi^{\ast}$ choose one action $\pi^{\ast}(s)$ for each state $s$, LP constraints imply that for any state $s$:
> $$
> V(s) \geq R(s, \pi^{\ast}(s)) + \gamma \sum_{s'} P(s'|s, \pi^{\ast}(s)) V(s')
> $$
> Let's write them in matrix form:
> $$
> V \geq R^{\pi^{\ast}} + \gamma P^{\pi^{\ast}} V
> $$
> while the $R^{\pi^{\ast}}$ is the reward vector under policy $\pi^{\ast}$, and $P^{\pi^{\ast}}$ is the transition matrix under policy $\pi^{\ast}$. We can rearrange the above inequality as:
> $$
> (I - \gamma P^{\pi^{\ast}}) V \geq R^{\pi^{\ast}}
> $$
> Since $\gamma < 1$, we can conclude that $I - \gamma P^{\pi^{\ast}}$ is invertible, and we can get its inverse as:
> $$
> (I - \gamma P^{\pi^{\ast}})^{-1} = \sum_{k=0}^{\infty} (\gamma P^{\pi^{\ast}})^k
> $$
> Obviously, the above inverse is a non-negative matrix. Thus, we can multiply both sides of the inequality by $(I - \gamma P^{\pi^{\ast}})^{-1}$ to get:
> $$
> V \geq (I - \gamma P^{\pi^{\ast}})^{-1} R^{\pi^{\ast}} = V^{\pi^{\ast}}
> $$
> Since $V^{\ast} \geq V^{\pi^{\ast}}$, we can conclude that $V \geq V^{\ast}$.
> So, we have shown that any feasible solution is greater than or equal to $V^{\ast}$, and the optimal value function $V^{\ast}$ is a feasible solution. Thus, the optimal solution to the linear programming problem is $V^{\ast}$.

Given we have the $V^{\ast}$, we can easily derive the optimal policy $\pi^{\ast}$ as:
$$
\pi^{\ast}(s) = \arg\max_{a} \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^{\ast}(s') \right]
$$

Here I asked claude to give me a simple explanation of the equivalence between this greedy like policy decision and the optimal policy. The formal proof may be need to use the fixed point theorem, but it is beyond the scope of this post. We just need to remember that the optimal policy can be derived from the optimal value function by choosing the action that maximizes the expected return.

## The Dual Linear Programming for MDP
The dual linear programming formulation for MDP is as follows:
$$
\begin{aligned}
\text{maximize} \quad & \sum_{s, a} \rho(s, a) R(s, a) \\\\
\text{subject to} \quad & \sum_{a} \rho(s', a) = \sum_{s, a} \rho(s, a) P(s'|s, a), \\\\
&\quad \forall s' \in S \\\\
& \rho(s, a) \geq 0, \quad \forall s \in S, a \in A
\end{aligned}
$$

If the initial state distribution $\mu(s) > 0$ for all $s \in S$, then let
$w(s) = \mu(s)$, the target function means the expected accumulated reward represented by the occupancy measure, and the constraints means the flow conservation constraints.

Assume that the optimal solution is $\rho^{\ast}(s, a)$, we can derive the optimal policy from **Theorem 2** as:
$$
\pi^{\ast}(s) = \frac{\rho^{\ast}(s, a)}{\sum_{a} \rho^{\ast}(s, a)}
$$

## Comparison between the Primal and Dual Linear Programming for MDP

| Dimension | Primal LP | Dual LP |
| --- | --- | --- |
| variable | state value function $V(s)$ | occupancy measure $\rho(s, a)$ |
| objective | minimize $\sum_{s} V(s)$ | maximize $\sum_{s, a} \rho(s, a) R(s, a)$ |
| constraints | Bellman optimality constraints | flow conservation constraints |
| explanation| state value|action frequency|


## Summary
At the beginning of this post, I tried to introduce the MDP-based Reinforcement learning, but I found that the solution of MDP takes a lot of space, so I just introduce the Bellman equation and the linear programming formulation for MDP. In the next post, I will introduce the value iteration and policy iteration algorithms for solving MDP, which are based on the Bellman equation.
