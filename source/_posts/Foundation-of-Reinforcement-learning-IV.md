---
title: Foundation of Reinforcement learning(IV)
date: 2026-05-18 12:18:13
description: "Model-free Reinforcement learning, study for the coming exam"
tags: 
  - Reinforcement learning
  - review notes
categories:
    - blog
---
## Introduction
In the previous posts, we have introduced the MDP and its solution. But in practice, we often do not have the simulation of the environment, which means we cannot directly apply our knowledge of MDP to solve the problem. There is indeed a method to simulate the environment, which is called model-based Reinforcement learning. However, in this post, we will focus on the model-free Reinforcement learning, which does not require the simulation of the environment and the construction of MDPs.

## Estimating Value Functions
In mode-based RL, value functions can be computed by DP methods as follows:
$$
\begin{aligned}
    V^{\pi}(s) &= E_{\pi} \left[R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \cdots | s_0 = s \right]  \\\\ &=
R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^{\pi}(s')
\end{aligned}
$$

However, in model-free RL, we cannot directly access the $P(s'|s, a)$ and $R(s, a)$, but we have some ways to estimate the value function from episodes of experience. 

> Why we estimate the value function? 
> Because we can use the value function to derive the optimal policy, which is our ultimate goal. Besides value function can help us to reuse historical experience to make better decisions in the future, which is the essence of Reinforcement learning.

Here is a graph to introduce some methods to estimate the value function:
<div style="text-align: center;">
  <img src="methods.png" alt="value estimation(Slide credit: David Silver)" width="400">
</div>

## Monte Carlo methods

**Target**: Learn $V^{\pi}$ from episodes of experience.

**Review**: accumulate reward function:

$$
G_t = R_{t} + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
$$

**Review**: value function is the expected return:
$$
V^{\pi}(s) = E_{\pi} \left[ G_t | s_t = s \right] \simeq \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^i
$$
The Monte Carlo method use empirical mean cumulative reward instead of expected return to estimate the value function.

### First-visit Monte Carlo method
The first-visit Monte Carlo method estimates the value function by averaging the returns following the first time a state is visited in an episode. The algorithm is as follows:
1. **Initialization**: 
    - For any $s \in S$,$V(s) \in \mathbb{R}$, $N(s) = 0$.
    - For any $s \in S$,$\text{returns}(s) = \emptyset$.
2. **Loop for each episode**:
    - Generate an episode following policy $\pi$: $S_0, A_0, R_1, S_1, A_1, R_2, \cdots, S_T$.
    - For t = T-1, T-2, ..., 0:
        + $G \leftarrow \gamma G + R_{t+1}$
        + If $s$ is the first time in the episode:
            * Append $G$ to $\text{returns}(s)$.
            * $V(s) \leftarrow \text{average}(\text{returns}(s))$.


The reason why we call it 'first-visit' is that we only update the value function for the first time we visit a state in an episode.Thus we can avoid the bias caused by multiple visits to the same state in an episode. However, this method may have high variance because it only uses one return for each state in an episode.

### Incremental Monte Carlo method
The first-visit Monte Carlo method takes a lot of memory to store the returns for each state, which is not efficient. The incremental Monte Carlo method uses an incremental update rule to estimate the value function without storing all the returns. The algorithm is as follows:
1. **Initialization**:
    - For any $s \in S$,$V(s) \in \mathbb{R}$, $N(s) = 0$, $G = 0$.
2. **Loop for each episode**:
    - Generate an episode following policy $\pi$: $S_0, A_0, R_1, S_1, A_1, R_2, \cdots, S_T$.
    - For t = T-1, T-2, ..., 0:
        + $G \leftarrow \gamma G + R_{t+1}$
        + If $s$ is the first time in the episode:
            * $N(s) \leftarrow N(s) + 1$
            * $V(s) \leftarrow V(s) + \frac{1}{N(s)} (G - V(s))$.

Interesting, online softmax also takes the same update rule as the incremental Monte Carlo method.Great job.


Besides, incremental MC provides more design space for us to tackle some problems in practice. For example, we can use a constant step size $\alpha$ instead of $\frac{1}{N(s)}$ to update the value function, which is called constant step size MC method. It is useful when the environment is non-stationary, which means the reward function and transition probability may change over time. In this case, we want to give more weight to recent returns than old returns, which can be achieved by using a constant step size $\alpha$:
$$
V(s) \leftarrow V(s) + \alpha (G - V(s))
$$


### Some properties of Monte Carlo methods
1. Monte Carlo methods are model-free, which means they do not require the knowledge of the environment's dynamics (transition probabilities and reward function).
2. Monte Carlo methods take the simpliest approach to estimate the value function, which is to average the returns following the policy. However, this method may have high variance because it only uses one return for each state in an episode.
3. One key to note is that Monte Carlo methods can only be applied to finite MDPs, which means the state space and action space must be finite.

### Importance sampling
Let's try to estimate a custom distribution $p(x)$ 's expectation.
$$
\begin{aligned}
    E_{x \sim p} [f(x)] &= \int f(x) p(x) dx \\\\ &= 
     \int f(x) \frac{p(x)}{q(x)} q(x) dx \\\\ &= 
    E_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right]
\end{aligned}
$$
Then we reassign the importance sampling weight $w(x) = \frac{p(x)}{q(x)}$, we can rewrite the above equation as:
$$
E_{x \sim p} [f(x)] = E_{x \sim q} \left[ f(x) w(x) \right]
$$

### off-policy Monte Carlo methods via Importance Sampling
We can use the cumulative reward function of policy $\mu$ to justify policy $\pi$, and then weight the cumulative reward function by the importance ratio between $\pi$ and $\mu$ to estimate the value function of policy $\pi$. The algorithm is as follows:

Every episode would be mutified by the importance sampling ratio:
$$
G_t^{\pi/\mu} = \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)} \frac{\pi(A_{t+1}|S_{t+1})}{\mu(A_{t+1}|S_{t+1})} \cdots \frac{\pi(A_{T-1}|S_{T-1})}{\mu(A_{T-1}|S_{T-1})} G_t
$$

So we then update the value function by:
$$
V(s) \leftarrow V(s) + \frac{1}{N(s)} (G_t^{\pi/\mu} - V(s))
$$

Sample by importance sampling will significantly increase the variance of the return, which is because the importance sampling ratio can be very large when $\pi$ and $\mu$ are very different. 


## Temporal-Difference Learning
Temporal-Difference (TD) is a method combining the MC method and DP method, which name comes from the fact that it uses the diff of estimated value function at two consecutive time steps to update the value function. There are two key ideas in TD learning: TD error and TD target.

For state value funtion $V$, after a transition from state $s$ to state $s'$ with reward $r$, the TD error is defined as:
$$
\delta = r + \gamma V(s') - V(s)
$$
The TD target is defined as:
$$
\hat{V} = r + \gamma V(s')
$$

As for the TD in Bellman expectation equation, the TD error is used in estimating the expect part.

### Some details of TD learning
The simpliest TD learning algorithm is called TD(0), which updates the value function by the TD error at each time step. The key equation of TD(0) is as follows:
$$
V(s) \leftarrow V(s) + \alpha \delta = V(s) + \alpha (r + \gamma V(s') - V(s))
$$

> Why we update like this?
> The Bellman expectation equation is rewritten as:
>
> $$
> E_{\pi} \left[ R_{t+1} + \gamma V^{\pi}(S_{t+1}) - V^{\pi}(S_t) | S_t = s \right] = 0
> $$
> That's all. We want to make the TD error as small as possible, which means we want to make the estimated value function as close as possible to the true value function. Thus, we can use the TD error to update the value function.

The TD method introduce the bootstrapping idea, which means we use the estimated value function to update the value function. This is different from the MC method, which uses the actual return to update the value function. The bootstrapping idea can significantly reduce the variance of the return, but it may introduce bias because we are using an estimated value function to update the value function.

### Contrast between TD and MC methods
They have the same goal: **Learn the value function from episodes of experience**. However, they have different approaches to achieve this goal. The MC method uses the actual return to update the value function, which can have high variance but no bias. The TD method uses the estimated value function to update the value function, which can have low variance but may introduce bias.

| TD method | MC method |
| --- | --- |
|update value function $V(s)$ like $V(s) \leftarrow V(s) + \alpha (r + \gamma V(s') - V(s))$|update value function $V(s)$ like $V(s) \leftarrow V(s) + \frac{1}{N(s)} (G_t - V(s))$|

The object of TD is $ R_t + \gamma V(s_{t+1})$, which is called TD target, while the object of MC is $G_t$, which is the actual return. The TD method's error is called TD error, which is defined as $\delta = r + \gamma V(s') - V(s)$, while the MC method's error is defined as $G_t - V(s)$.

### The strengths and limitations of TD learning and MC learning

TD method can learn until the end of an episode:
- After each step in an episode, TD method can update the value function use the former value function, which means it can learn until the end of an episode. However, MC method can only update the value function after the end of an episode, which means it cannot learn until the end of an episode.
- TD method can learn from incomplete episodes, which means it can learn from episodes that are not terminated. However, MC method can only learn from complete episodes, which means it cannot learn from episodes that are not terminated.

### Tradeoff between bias and variance

| | Estimator | Bias | Variance |
|---|---|---|---|
| MC | $G_t$ | Unbiased: $E[G_t] = V^{\pi}(s)$ | Higher |
| TD (real) | $R_{t+1} + \gamma V^{\pi}(S_{t+1})$ | Unbiased: $E[R_{t+1} + \gamma V^{\pi}(S_{t+1})] = V^{\pi}(s)$ | Lower |
| TD (actual) | $R_{t+1} + \gamma V(S_{t+1})$ | **Biased**: $E[R_{t+1} + \gamma V(S_{t+1})]$ ≠ $V^{\pi}(s)$ | Lower |

> **Note**: The real TD target uses the true $V^{\pi}$, which is unknown in practice. The actual TD target uses the current estimate $V$, introducing bias. Despite the bias, TD typically has lower variance than MC because it bootstraps from a single step rather than a full trajectory.

## Multi-step TD learning
The TD(0) method only uses the immediate reward and the estimated value of the next state to update the value function, which may not be sufficient to capture the long-term dependencies in the environment. The multi-step TD learning method uses the rewards and estimated values of multiple future states to update the value function, which can better capture the long-term dependencies in the environment. We will introduce it by leading into n-step cumulate reward function and n-step TD target.

### n-step cumulate reward function
Consider the following n-step cumulate reward function:
$$
G_t^{(n)} = R_{t} + \gamma R_{t+1} + \cdots + \gamma^{n-1} R_{t+n - 1} + \gamma^n V(S_{t+n})
$$
It seems make sense to use the n-step cumulate reward function to update the value function, which is called n-step TD learning. The key equation of n-step TD learning is as follows:
$$
V(s) \leftarrow V(s) + \alpha (G_t^{(n)} - V(s))
$$

### n-step mean cumulate reward function
Can we take up the information of different n-step cumulate reward function to update the value function? The answer is yes. 
We can use a weighted average of different n-step cumulate reward functions to update the value function, which is called n-step mean TD learning. The key weight figure of weighted average is as follows:
<div style="text-align: center;">
  <img src="weight.png" alt="lambda weight" width="300">
</div>

So the n-step mean cumulate reward function is defined as:
$$
G_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
$$
Then we can update the value function by:
$$
V(s) \leftarrow V(s) + \alpha (G_t^{\lambda} - V(s))
$$

This is called TD($\lambda$) method, which is a generalization of TD(0) and MC methods. When $\lambda = 0$, TD($\lambda$) reduces to TD(0) method, and when $\lambda = 1$, TD($\lambda$) reduces to MC method. Thus, by adjusting the value of $\lambda$, we can control the bias-variance tradeoff in the estimation of the value function.

### Conclusion about TD(λ) method

- Unless the $lambda$ is 0, TD($\lambda$) mothod is unbiased, because it's a weighted average of unbiased n-step TD targets.
- The variance of TD($\lambda$) method is lower than that of MC method, because it uses bootstrapping to update the value function, which can reduce the variance of the return. However, the variance of TD($\lambda$) method is higher than that of TD(0) method, because it uses more rewards and estimated values to update the value function, which can increase the variance of the return.

    $$
    \text{Var}(aX + bY) = a^2 \text{Var}(X) + b^2 \text{Var}(Y)
    $$
- Empirically $\lambda$ is not quite commom because fast credit assignment for a given action is preferred. So MC or TD(0) is more commonly used in practice. However, TD($\lambda$) method can be useful when we want to balance the bias-variance tradeoff in the estimation of the value function, which can be achieved by adjusting the value of $\lambda$.

So TD($\lambda$) use $\lambda$ as variable while n-step TD use $n$ as variable. TD($\lambda$) is a generalization of n-step TD, which can be seen as a weighted average of infin-step TD targets. By adjusting the value of $\lambda$, we can control the bias-variance tradeoff in the estimation of the value function, which can be useful in practice when we want to balance the bias and variance in the estimation of the value function.


## Conclusion
In this post, we have introduced the model-free Reinforcement learning, which does not require the simulation of the environment and the construction of MDPs. We have introduced two methods to estimate the value function from episodes of experience: Monte Carlo methods and Temporal-Difference learning. We have also introduced the n-step TD learning method, which uses the rewards and estimated values of multiple future states to update the value function, which can better capture the long-term dependencies in the environment. Finally, we have discussed the bias-variance tradeoff in the estimation of the value function, which can be controlled by adjusting the value of $\lambda$ in TD($\lambda$) method.