---
title: "对 Looped Transformer 的简要学习笔记"
date: 2026-09-06 20:30:47
description: "Looped / Recurrent Transformer 学习笔记：从 Universal Transformer、CoTFormer 到 Mixture-of-Recursions 的 halting 机制与 KV 管理方案梳理，以及从 inference 视角对 per-token R 与 continuous looping 的思考。"
tags:
  - Learning note
  - model architecture
  - LLM inference
categories:
  - readings
---

最近有关GPT6使用了recurrent transformer即looped transformer架构的小道消息沸沸扬扬，一些过去的工作也逐步进入了大众视野。正值完美的周末学习时间，于是调研一下相关工作以及从推理角度分析其是怎么进行serving，很多东西都是个人的一些看法，也是在很短的时间内学习然后成篇的，有错误认知希望大家谅解并感谢纠正🥹，欢迎大家交流ww。

## Looped transformer arch

众所周知传统的模型的层数均为固定的层数，activation流经各层之后就变成了美味的token😋，而looped transformer提出了在模型层数上进行重复来进行scale up的新的可行域。其主要的结构也就不过差不多是：
$$
f(x) = C_{\theta_C} \circ F_{\theta_R}^{R} \circ P_{\theta_P} \circ E(x)
$$
在古早的阶段或者受限的情况，$R$由serving侧决定。某些设想是由session来决定$R$。但是参考MoE的发展，一个可以去猜想的未来发展是per token R。我也不知我为什么会这么去想，大概之前看到有人说之前的moe是per request并不好，现在都是per token，所以大胆地去设想per token R是一个可预见的未来。

在大致弄懂了looped transformer的结构之后，我们便选择去查看相关的论文，看一眼现在大家是如何做的。

### Universal Transformer

最古老的文章大概是ICLR 2019的Universal Transformer，其中提出了复用transformer block的最原始的想法。其使用的ACT halting大致如下：
$$
p_i^t = \sigma(w^\top h_i^{(t)} + b) \\
$$

$$
R_i = \text{min}\{t:\sum_{\lambda = 1}^t p_i^{(\lambda)} \geq 1 - \epsilon\}
$$

看上去很依赖我们的$\sigma$函数。而且很大的一个问题是，仅仅考虑prefill阶段的一个token序列
$$
L = \{s_1, s_2, ... s_n\}
$$
在固定R下我们知道Attn层可以有固定映射的KV，但是若是R不对齐，那后续token的$KV$该如何对齐前文早停的token呢？

观察UT的做法，其采用将已halt的token冻结，这显然在算法和infra方面都不优雅：

1. 算法层次上是一个mixed depth representation 不优美。
2. infra层次上相当于所有的或者一个block内的depth为$\text{max}(R_i)$，也不是一个优美的方案。

### Sparse Universal Transformer

在EMNLP 2023上，我们看到了一个叫做Sparse UT的东西。虽然其叫Sparse的原因大概率是因为他在UT中引入了MoE和对Attn进行sparse化，但是其也对ACT halting做出了一些改变。看了一眼，其对Attn进行sparse化也有点小说法，但是不影响整体结构，我们还是主要分析halting。

SUT重新解释了halt的数学含义，认为我们需要去预测：
$$
P(R_t = l | R_t \geq l) = \alpha_l^{(t)}\prod_{l' < l}(1 - \hat{\alpha}_{l'}^{(t)})
$$
其中：
$$
\hat{\alpha_l}^{(t)} = \text{halt}(h_l^{(t)})
$$
然后就可以去自然地去定义expect $R_t$ ：

既然：
$$
P(R_t = l) = \alpha_l^{(t)}
$$
则：
$$
\mathbb{E}[R_t] = \sum_{l} l \alpha_l^{(t)}
$$
同样他对刚刚我们的mixed depth KV也有新的做法：

其维护当前第l层state $h_l^{(t)}$ 和一个供观察的state $s_l^{(t)}$, 其中$s_l^{(t)}$的定义比较有道理（x
$$
s_l^{(t)}
=
\left(
1-\sum_{l'=1}^{l-1}\alpha_{l'}^{(t)}
\right)h_l^{(t)}
+
\sum_{l'=1}^{l-1}\alpha_{l'}^{(t)}h_{l'}^{(t)}
$$
我们结合Attn的结构具体分析，其直观表示即为：
$$
a_l^{(t)}
=
\operatorname{Attention}
\left(
\underbrace{h_{l-1}^{(t)}}_{Q},
\underbrace{S_{l-1}}_{K},
\underbrace{S_{l-1}}_{V}
\right).
$$

$$
S_l
=
\left\{
s_l^{(1)},
s_l^{(2)},
\ldots,
s_l^{(T)}
\right\}.
$$

可见他在UT上面套了一层壳，但是如果我们真的去思考$R_j > R_i$时我们还是在做frozen state, 几乎没有改变。

通过对这两篇论文的分析，我们发现其均是在runtime里动态决定当前token的R，而非使用一个prediction head在跑 L 层前决定，由此也造成很大的infra难题，我们干脆把动态决定的称作online halting，提前决定的称作upfront depth prediction（原谅笔者随便取名ww）。但是提前决定应该在算法角度严格劣于online halting。

### CoTFormer

在ICLR 2025上，我们看到了一篇叫做CoTFormer的文章，这篇文章的作者们着重解决了一下mixed depth state问题。大致想法如下：

正常来讲，一个UT的思路大致如下：
$$
x_t^{(r + 1)} = B_{\theta}(x_t^{(r)}, X_{<t}^{(r)})
$$
而且，如果某一个历史token早停，那便只能定义$x_j^{(r)} = x_j^{R_j}, r > R_j$.

但是，在某些理解上，对于一个token的反复重复应当被视为模型的思考过程，而模型应当对此关注而非认为是不同depth而置之不理，也就是说其将上式改为了：
$$
x_t^{(r+1)}
=
B\left(
x_t^{(r)},
\left[
X_{<t}^{(r)},
X_{<t}^{(r-1)},
\ldots,
X_{<t}^{(0)}
\right]
\right)
$$
我认为这是现代我们认为looped layers实际上是latent CoT的雏形。这篇文章也很好地解决了mixed depth state问题，但是其缺点也显而易见，相对来说上下文变长了，但是相应地cot tokens也会变少，为了控制总的FLOPS，CoTFormer对halt机制也进行了一些调整。

对于一个token j，在其跑完第$r$次之后，对下文有一个learned vector $e^{(r)}$，然后我们计算出核心指标$s_j^{(r)} = \sigma(e^{(r)\top} x_j^{(r)})$，然后最关键的是我们在每一轮之前会给出一个capacity $c_r \in [0, 1]$ 下一轮按照topk只允许$k_r = c_rS$个token继续。这样我们对于token继续的轮数有了巨大可控性，runtime过程也可以进行很多事情做了✌️。

但是引用数量好惨淡（x

### Inner Thinking Transformer

在ACL 2025上，我们高兴地读到了一篇故事会Inner Thinking Transformer。作者们去观察GPT-2的attention gradient nuclear norm，发现不同token的gradient不太一样，作者提出了一种解释：有一些token太难了所以应该多重复几次。我们可以直接理解为在部分layer上做一部分token的额外重复计算工作。

值得注意的是，作者们提出的重复计算并非只取最后一轮，而是采用了一个叫做Residual Thinking的操作：
$$
x^{(t)}
=
\sum_{i=1}^{t}
f\left(x^{(i-1)}\right)\odot \phi^{(i)}
$$
RTC 在数学形式上同样组合多个 recurrent step 的 representation，因此与 SUT 的 expected halted state 有表面上的相似性；但二者语义不同：SUT 的权重来自 halting distribution，而 RTC 的 $\phi^{(i)}$ 是无概率约束的 learnable step embedding。与前几篇不同，其对应Looped Transformer里的$R$的东西也很难算是由token完全决定的。没有去考虑KV的mix depth问题。感觉算是一个从新的角度重新发现了Looped transformer，但是似乎很多东西都没有搞得很明白。

### Mixture-of-Recursions

NeurIPS 2025百引论文Mixture of Recursions给我们的Looped Transformer问题做了很多里程碑式的解释和补全。

该工作提出了两条router路径：

1. Expert Choice, 同CoTFormer类似，该路径采取hierarchical filtering的topk selection，是一种online halting路径。
2. Token Choice，这是与MoE router类似的想法，也是我一开始自然而然的思路，大致就是利用一个router在进入recursive block之前直接predict R，显然是一种upfront路径。

作者们也测试了这两种办法的效果，Expert Choice准确率优于Token Choice，也符合预期。

另一个很有意思的问题，哎我之前都没注意到，如果我们采用topk selection的话，实际上是违反causal原则的，然后作者用了一些手段稳定了训练，这是算法人的工作，我们就不研究了（x。

对于我们之前一直关注的KV mixed depth问题，MoR同样给出了两种路径：

1. Recursion KV caching，$\mathcal{A}_r = \{\, i : R_i \ge r \,\}.$ 那么只计算和缓存： $\left\{ K_i^{(r)}, V_i^{(r)} : i \in \mathcal{A}_r \right\}$ 并且第 $r$ 轮 query 只允许 attend 当前 recursion 仍然 active 的 token： $Q_i^{(r)} \operatorname{Attn} \left( K_j^{(r)}, V_j^{(r)} \right), \qquad j \in \mathcal{A}_r.$ 简单粗暴。
2. Recursive KV Sharing，$Q_{\mathcal{A}_r}^{(r)} \leftrightarrow K_{1:S}^{(1)}.$ 让所有深层都关注一遍浅层

显然这两个各有优劣，实际使用需要hybrid。

### Huginn

在2025年2月，有一个3.5B的recurrent depth模型验证了loop transformer 可以被scale up。

不同于UT的结构：
$$
h^{(r + 1)} = F(h^{(r)})
$$
该模型的结构是：
$$
s^{(r + 1)} = F(s^{(r)}, e)
$$
其中$e$是prologue  layer之后得到的表示。其训练方法是每一个sequence随机采样一个$r \sim \Lambda.$

其中：
$$
\tau
\sim
\mathcal{N}
\left(
\log \bar{r}
-
\frac{1}{2}\sigma^2,
\sigma
\right)
$$

$$
r
\sim
\operatorname{Poisson}\left(e^{\tau}\right)+1.
$$

$$
\bar{r}=32.
$$

在$R$的选取上，该方案也是runtime的online router，通过计算token前后的KV散度来实现$R$的选取。对于mixed depth KV，其同样采取了frozen，但是训练成功了，我个人认为与其采用的KL散度有关，相近的token的R是相通的，所以自然frozen的可实现性是比较高的。

### Ouro

近来的“韩国股市”在模型架构领域的定型文中，经常出现字节looped transformer，也就是这一篇。不过说是seed只发失败的东西，但是又听说llm行业重要的是什么不能做，于是我们来批判性地欣赏seed是怎么做的。

感觉今下午读到现在，越来越感觉looped transformer实际上跟dit很像，一轮一轮地扩散生成，像ouro也是每一个loop实际上都会装一个LM Head。在loss函数设置上，ouro明确地要求模型学习diverse R而非一个min FLOPS。在推理上，仍然使用online depth allocation。

对于KV问题，他们做了一个很有趣的消融，用全部vs只用第一个vs只用最后一个vsmean，然后他们发现使用last几乎没有影响。但是这是在R = 4下做的，实际情况或许会有很多不一样的R，这时还能继续这样做吗？而且其在测量R = 5,6,7,8时，出现了benchmark下降的情况，感觉效果也不太好，Ouro更像是以工程量著称而非算法。

### Think-at-Hard

这是一篇投了ICML 2026论文，看上去是对机制的进一步验证。详细说来，作者发现存在latent overthinking现象。他们似乎去做了一堆实验，来看R = 1 / R = 2的一些token现象，证明了overthinking会降低表现。这显然预示着我们per token R是未来的必然。但是这篇论文的大头似乎是在解决attn上。

**Duo-Causal Attention** 是这篇文章的核心卖点。大致来讲，Think-at-Hard的作者认为传统的attention只会关注之前的token，也即$i < j$，现在我们还要再加一条约束，即query $(i, d)$可以attend：
$$
(j, k) \space\space \text{iff} \space\space j \leq i, k\leq d
$$
 读到现在的读者或许会感觉其与CoTformer非常相像，但是值得注意的是TaH的$R2 >> R1$ 且不共享参数，因此还是有一些不同的。感觉这个kernel还是有点意思，算子应该写着挺爽的（x

此外，基本思路比较容易，就是一个修错的token的思路，也是采用着online halt。

### Anira

2026年2月份release到arXiv 上的文章Understanding Dynamic Compute Allocation in Recurrent Transformers做了一个解答我一直的疑问的实验——提前预测R vs online halting哪个比较好。

继续验证了online在算法角度上是优于early predictor，但是kv仍然采用了frozen的做法。虽然我们感觉到他跟那篇MoR实验很像，但是这篇文章提出了一个更新的观点：这两种方式适合解决不同的问题。

> Early allocation relies more on static structural cues and online halting tracks algorithmic execution state.

这篇文章主要是在研究R到底学了什么的。

### Adaptive Latent CoT

一些已有架构(Duo attn) + 更多的训练技巧

### AdaPonderLM

Diffusion online router LLM with freezen KV

### PonderLM-3

upfront router with CoTformer like KV and training tricks

上面这三篇均是2026.3发到arxiv上的，核心作者也都一样，认为是一些排列组合（x

此外还有很多文章比如说Adaptive Recurrence、Per token Fixed Point Convergence、RecurTrace等等，不过看到现在，我们可以看到这些looped transformer都基本上大差不差。现在大家基本上都围绕着$R$是online，KV该怎么管理去探索，基本的推理范式感觉不会发生太大变动，但与传统的fixed layer model的推理还是有所不同，于是进一步仔细思考一下。

## Inference 

感觉looped transformer逐渐有一些diffusion model的味道了，出一个token草稿然后不断修正，确实感觉让人耳目一新。但是具体分析我们该如何推理，似乎现在的主流技术路线还是存在一些分歧。

但是，经过我们上文对这些工作的分析，无非现在就只有两个问题发生了改变：

1. 历史**KV**用到什么程度
   + Freeze/ Latest
   + 每一个depth用自己当前depth的token序列
   + Canonical KV
   + All depth KV
2. $R$的选取时机
   + Upfront 可预测
   + Online
   + Per-step routing

这么看来，首先对于 KV 的问题，我感觉算法侧最后大概率还是收敛到一个比较稳定的方案上。虽然现在不同文章里面会看到 Freeze/Latest、每一个 depth 只使用自己当前 depth 的 token、Canonical KV、All-depth KV 这些不同做法，但是从serving 的角度看，alldepth显然会给系统带来比较大的额外负担。对比正常 Transformer 一个历史 token 在一层里面只需要对应一份 KV，如果把每一个 recurrent depth 都作为独立 memory 保存，那么 KV 的数量实际上会跟 $\sum_i R_i$ 成正比；平均一个 token loop 三四次，就相当于把原来一份 KV 变成三四份，这对于 decode 这种本身就很容易 memory-bound 的 workload 显然不是一个很优雅的设计。或许大家会证明一件事情，就是latent CoT与显式CoT占用的KV cache会接近不会发生太大改变。在KV cache上感觉不会变化太大，即使我们使用了all depth也无非就是扩大加载的问题（怎么感觉p5k要暴毙的节奏x）。

对于 $R$ 来说，decode 侧不一定是一个特别大的问题。传统 continuous batching 本来就在不断把不同 request 当前需要执行的 token work 拼成一个 batch，而现在只不过把原来一个 token 的一次 forward 再拆成若干个 recurrent step。对于共享参数的 recurrent core，不同 request 即使现在处于不同的 loop depth，本质上跑的仍然是同一个 $F_\theta$，所以完全可以把它们重新聚合到一个 batch 里。比如 request A 现在在第一轮，B 在第三轮，C 在第二轮，只要它们都要进入同一个 recurrent block，那么实际上就可以一起执行；执行完以后，已经 halt 的 token 进入 coda 或者 LM head，仍然需要继续的 token 把自己的 recurrent depth 加一以后重新丢回 ready queue。换句话说，原来的 continuous batching 可以进一步拆成一种 continuous looping，scheduler 管理的基本单位从“某一个 request 的下一个 token”变成“某一个 request 的当前 token 在某一个 recurrent depth 上的一次 work item”。尤其是对于 upfront $R$ 的情况，当前 token 在真正进入 loop 之前就已经知道需要执行多少次，那么 scheduler 甚至可以提前知道它未来还会占用多少次 recurrent block，从 batch formation、KV reservation 到 CUDA Graph 的选择都可以做得比较准确。Online $R$ 会稍微麻烦一些，因为每跑完一轮才能知道下一轮是否还存在，不过这和今天各种 continuous batching scheduler 里面不断有 request finished / resumed / preempted 的状态变化相比，我感觉仍然属于可以管理的动态性，并没有改变整个 decode pipeline 的基本形态。

但是不得不忽视的是 prelude、recurrent core 和 coda 之间固定开销的问题。很多 looped transformer 并不是整模型所有层全部共享，而是类似 Huginn 一样有一个 prelude，中间一小段 recurrent core，再接一个 coda。这样 continuous looping 实际上只能非常自然地发生在 recurrent core 内部，因为不同 depth 的 token 跑的是同一组参数；prelude 和 coda 还是各自固定的一段网络。所以最后的 scheduler 可能不再只有一个普通 decode queue，而是会形成 prelude queue、recurrent queue 和 coda queue 三个阶段。recurrent queue 本身很容易 batch，但是如果某个时间点大量 token 同时从 recurrent core 退出进入 coda，或者有大量新 token 同时从 prelude 进入 recurrent core，就有可能出现阶段之间的负载不均衡，感觉真正要优化的是不同 stage 之间的 batch formation 和固定开销。

对于 prefill 阶段，传统 Transformer 的 prefill 之所以效率高，一个很重要的原因就是整个 prompt 上的 token 可以组成很大的 dense GEMM 和 FlashAttention workload；比如一个几千 token 的 sequence，本质上给了 GPU 一个 $M=S$ 的大矩阵计算。但是有了 per-token $R$ 以后，如果不同 token 在不同 recurrent depth 退出，那么最自然的执行方式应该还是按照 depth 一轮一轮地做，而不是每个 token 单独把自己的 loop 跑完。也就是说第一轮所有 token 都执行，第二轮只执行 $R_i\ge2$ 的 token，第三轮只执行 $R_i\ge3$ 的 token，这样 sequence 维的并行仍然存在，理论总计算量也能够从 $S R_{\max}$ 降到 $\sum_i R_i$。问题在于 active token 数会随着 depth 不断下降，原来一个很大的 workload 会从 $S$ 逐渐变成 $A_2,A_3,\ldots$，后面几轮可能很快退化成中小规模甚至非常稀疏的计算。这个时候算法上省掉的 FLOPs 能不能真正兑现成 latency，就很值得怀疑了，因为 Tensor Core utilization、occupancy、kernel launch overhead 以及 token compaction 的成本都会开始变得重要。

Upfront $R$ 在 prefill 这里感觉会比在 decode 侧更有意义一些。因为如果 recurrent computation 开始之前就已经知道整个 prompt 上每个 token 的 $R_i$，那所有 depth 的 active set 其实一开始就都知道了，我们完全可以提前把每一层需要计算哪些 token、对应的 GEMM shape 多大、KV 要怎么放全部规划好，然后一次性按照这个 schedule 往下执行。甚至可以进一步做 depth bucketing，把 $R$ 相近的 token 尽量组织成规则一些的 workload。Online $R$ 就没有这么优雅，每一轮必须先把当前 active token 跑完，再经过 router 或 convergence criterion 才知道下一轮有哪些 token，因此每一轮都可能需要重新 compact、重新生成 metadata、重新选择 kernel。不过我倒不觉得 router 本身会成为 prefill 的主要瓶颈，因为 prompt 足够长的时候，这些固定开销应该可以被大规模计算分摊；真正的问题感觉还是随着 active set 不断缩小以后，GPU workload 本身越来越不规则。

KV 的选择在 prefill 里面也会直接决定这种 sparsity 最后的样子。如果是 Freeze/Latest 或者 Canonical KV，那么后面第 $r$ 轮虽然只有一部分 token 继续产生 query，但是这些 query 仍然可以对完整 prompt 的 KV 做 attention，所以 workload 更像是 Q 的长度不断缩小，而 KV 长度保持为 $S$。这样至少 KV 侧还是一个比较规则的完整 sequence，系统实现相对容易，只不过省掉 recurrent computation 并不能同比例省掉 attention 里面读取历史 KV 的开销。如果是 MoR 那种 same-depth KV，那么第 $r$ 轮的 query 和 KV 都只来自当前 active token，理论计算复杂度可以随着 active set 的平方下降，算法上似乎会更优雅一些，但是这些 active token 在原序列里面通常并不是连续的，比如可能只剩第 2、5、8、11 个 token，因此 attention 已经不再是一个普通的连续 causal sequence，而是带原始 position metadata 的 ragged causal attention，kernel 侧会变复杂。All-depth KV 则更特殊，因为随着 recurrent depth 增加，query 数量在不断减少，但是之前产生过的 latent KV 却一直在累积，最后形成一种 Q 越来越短、KV 越来越长的 workload；从推理角度看，这应该是现在几种方案里面最难的一种。

所以到这里我感觉 decode 和 prefill 的主要矛盾其实可以区分得比较清楚。Decode 的问题更多是 scheduling，即怎么把不同 request、不同 recurrent depth 上的很小的 work item 重新 continuous batch 成大 batch；而 prefill 的问题更多是 shape，即怎样在 active token 数不断下降的情况下，不让原本很高效的 dense prefill 退化成一堆 utilization 很差的小矩阵计算。前者主要是在已有 continuous batching 上继续细分 scheduler，后者可能真正需要一些新的 kernel/runtime co-design。

甚至进一步想，我感觉最后算法侧给出的最优 $R_i$ 和系统侧真正想要的 $R_i$ 可能不会一样的。算法可能希望每一个 token 都非常精确地选择 1、2、3、4、5、6 次 loop，但是从 GPU 角度看，这样会产生很多不同大小的 active set 和非常碎的 execution shape。反过来，如果把 $R_i$ 强制量化成少数几个 bucket，比如只允许 1、2、4、8，那么虽然某些 token 会多算几轮，理论 FLOPs 反而增加，但是整个 prefill 可以维持更大的 batch、更少的 shape、更稳定的 kernel 配置，最后 wall-clock 反而可能更低。也就是说这类模型真正进入推理系统以后，我们可能不应该只优化 $\sum_i R_i$，而应该直接把 GPU execution cost 放进 $R$ 的设计里面。算法上最省 FLOPs 的 routing policy 不一定是系统上最快的 routing policy，这一点感觉可能会成为 looped transformer 真正落到 inference 上以后比较核心的问题。

OK今天的周末学习就到这里，感觉自己表达了很多自己的想法，或许都很有问题，菜菜勿喷呜呜🥹