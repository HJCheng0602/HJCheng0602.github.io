---
title: "尝试对Kimi KDA的数学推导&算子实现分析"
date: 2026-08-14 00:00:00
description: "对Kimi KDA attention的数学推导和flash KDA v1 kernel实现分析"
tags:
  - KDA
  - linear attention
  - kernel optimization
categories:
  - practice
---

## prologue

kimi KDA是kimi k3所使用的kernel, 但远早于k3发布(2026.2.17), 囿于笔者今年3月份才开始探究ai & ai infra, 且近来事务繁忙, 一直都没有认真阅读其论文和公式. Kimi 发布K3 report的时候, 也是对着K3的公式一脸茫然. 读剑林老师的博客的时候, 对模型结构的数学设计感到无比优雅, 心向往之. 恰好笔者近来有较多空闲, 于是便去研读了一遍KDA 的论文和一些网上的资料, 对一些数学推导和算子实现便成为如下笔记, 希望大家原谅笔者的数学水平和难以评价的表达.

## mainloop

直接进入正题, kimi k3的关键核心式如下(笔者懒惰懒得花功夫排版了, 大家只需要知道$o_t=q_ts_t\in\mathbb{R}^{1,V}$就好:

$$
s_t=\operatorname{diag}(\lambda_t)s_{t-1}+\beta_t\cdot k_t^{\top}\left(v_t-k_t\operatorname{diag}(\lambda_t)s_{t-1}\right)\in\mathbb{R}^{K\times V}\tag{1}
$$

即:

$$
s_t=\operatorname{diag}(\lambda_t)s_{t-1}+\beta_t\cdot k_t^{\top}v_t-\beta_t\cdot k_t^{\top}k_t\operatorname{diag}(\lambda_t)s_{t-1}\tag{2}
$$

发现$k_t^{\top}k_t$为$K\times K$方阵且可独立计算:

$$
s_t=\left(1-\beta_tk_t^{\top}k_t\right)\operatorname{diag}(\lambda_t)s_{t-1}+\beta_t\cdot k_t^{\top}v_t\tag{3}
$$

对于每一个chunk, 我们给予一个$S$作为$s_0$

则此时对于每一个$s_t$, 我们尝试仅根据$S$、$q$、$k$计算$s_t$

为了方便展开, 我们令

$$
\begin{aligned}
D_t&=\left(1-\beta_tk_t^{\top}k_t\right)\operatorname{diag}(\lambda_t)\in\mathbb{R}^{K\times K}\\
C_t&=\beta_t\cdot k_t^{\top}v_t\in\mathbb{R}^{K\times V}
\end{aligned}\tag{4}
$$

则:

$$
s_t=D_ts_{t-1}+\beta_t\cdot k_t^{\top}v_t\tag{5}
$$

认为现在已经很明朗了, 尝试继续推进:

$$
s_t=(D_tD_{t-1}\ldots D_1)s_0+\sum_{j=1}^{t}(D_tD_{t-1}\ldots D_{j+1})C_j\tag{6}
$$

对Infra工作而言,

单个计算$D_t$是无依赖的, 但是考虑到矩阵乘没有交换性, 因此该实现并不优雅, 且若考虑到显存更是毫无linear attention本该具有的优势.

于是考虑另外的方向:

我们重新出发:

$$
s_t=L_ts_{t-1}+\beta_tk_t^{\top}(v_t-k_tL_ts_{t-1}),L_t=\operatorname{diag}(\lambda_t)\tag{7}
$$

发现此时$L_t$的累乘非常好处理, 于是进行如下定义:

$$
P_t=L_tL_{t-1}\ldots L_1=\operatorname{diag}\left(\prod_{r=1}^{t}\lambda_r\right)\tag{8}
$$

这里我们再令:

$$
g_t=\sum_{r=1}^{t}\log\lambda_r\tag{9}
$$

则:

$$
P_t=\operatorname{diag}(e^{g_t})\tag{10}
$$

然后于是就发生了这样的代换:

$$
\hat{s}_t=P_t^{-1}s_t\tag{11}
$$

代入原式:

$$
P_t\hat{s}_t=L_tP_{t-1}\hat{s}_{t-1}+\beta_tk_t^{\top}\left(v_t-k_tL_tP_{t-1}^{-1}\hat{s}_{t-1}\right)\tag{12}
$$

进一步化简:

$$
\hat{s}_t=\hat{s}_{t-1}+\beta_tP_t^{-1}k_t^{\top}\left(v_t-k_tP_t\hat{s}_{t-1}\right)\tag{13}
$$

之后似乎陷入了一个困境, $\hat{s}_{t-1}$在括号里很难拿出来, 由此两个$P$也无法进行合并.

但是注意到:

$$
(P_t^{-1}k_t^{\top})\ and\ (k_tP_t)\tag{14}
$$

这两个之间似乎满足某种神奇的对偶关系.且我们更能发现$P_t$为对角阵

于是不妨这样定义:

$$
k_t^{+}=k_t\operatorname{diag}(e^{g_t})\tag{15}
$$

$$
k_t^{-}=k_t\operatorname{diag}(e^{-g_t})\tag{16}
$$

于是, 我们的递推式很好地变成了!:

$$
\hat{s}_t=\hat{s}_{t-1}+\beta_t(k_t^{-})^{\top}\left(v_t-k_t^{+}\hat{s}_{t-1}\right)\tag{17}
$$

现在不妨这样定义:

$$
\hat{v}_t=v_t-k_t^{+}\hat{s}_{t-1}\tag{18}
$$

那我们现在的递推式就是:

$$
\hat{s}_t=\hat{s}_{t-1}+\beta_t(k_t^{-})^{\top}\hat{v}_t\tag{19}
$$

那现在我们便很方便地写出一个chunk内的状态:

$$
\hat{s}_t=S+\sum_{j\le t}\beta_j(k_j^{-})^{\top}\hat{v}_j\tag{20}
$$

代回$\hat{v}_t$:

$$
\begin{aligned}
\hat{v}_t&=v_t-k_t^{+}\hat{s}_{t-1}\\
&=v_t-k_t^{+}S-\sum_{j\le t-1}\beta_jk_t^{+}(k_j^{-})^{\top}\hat{v}_j
\end{aligned}\tag{21}
$$

诶嘿, 这步代换堪称神来之笔, 我们得到了我们想要的标量$k^{+}(k^{-})^{\top}$

于是很自然地, 我们定义:

$$
A_{tj}=\begin{cases}
\beta_jk_t^{+}(k_j^{-})^{\top},&j<t,\\
0,&j\ge t.
\end{cases}\tag{22}
$$

然后看上去目前$A$天然便是下三角矩阵:

$$
A=\begin{bmatrix}
0&0&0&\cdots\\
A_{21}&0&0&\cdots\\
A_{31}&A_{32}&0&\cdots\\
\vdots&&&\ddots
\end{bmatrix}.\tag{23}
$$

于是, 我们便得到了一个很优美的方程:

$$
\hat{v}_t+\sum_{j<t}A_{tj}\hat{v}_j=v_t-k_t^{+}S\tag{24}
$$

即为:

$$
(I+A)\hat{V}=V-K^{+}S\tag{25}
$$

嗯很好地, 我们现在只需要解出$\hat{V}$即可, 鉴于$A\in\mathbb{R}^{C\times C}$是下三角矩阵, 有引理:

$$
(I-A)\cdot(1+A+A^2+\cdots+A^{C-1})=I\tag{26}
$$

显然$I+A$的逆矩阵是有解析解而且比较可求的, 记为:

$$
T=(I+A)^{-1}\tag{27}
$$

则:

$$
\hat{V}=T(V-K^{+}S)\tag{28}
$$

之前我们有这样的式子:

$$
\hat{s}_t=\hat{s}_{t-1}+\beta_t(k_t^{-})^{\top}\hat{v}_t\tag{29}
$$

现在将整个chunk展开:

$$
\hat{s}_t=S+\sum_{i=1}^{t}\beta_i(k_i^{-})^{\top}\hat{v}_i\tag{30}
$$

定义:

$$
K^{-}=\begin{bmatrix}
(k_1^{-})\\
\vdots\\
(k_C^{-})
\end{bmatrix}\in\mathbb{R}^{C\times K}\tag{31}
$$

以及:

$$
B_{\beta}=\operatorname{diag}(\beta_1,\ldots,\beta_C)\tag{32}
$$

则:

$$
\hat{s}_{out}=S+(K^{-})^{\top}B_{\beta}\hat{V}\tag{33}
$$

注意我们现在写的公式是对于token C处的展开, 这样我们就得到了下一个chunk的S

现在的核心问题转而变成了我们如何得到chunk内每一个token的$O$

从第(30)式出发固然是一个很好的想法, 但是显然会产生一个prefixsum的东西, 然而大多数情况下不需要我们显式地得到$s$, 而且$s$如果把所有的均算出来之后, 其为一个$C\times K\times V$的一个张量, linear attention的优势何在!?

于是, 我们选择直接将$q_t$与$s_t$相乘:

$$
\begin{aligned}
o_t&=q_t^{+}\hat{s}_t\\
&=q_t^{+}S+\sum_{j\le t}\beta_jq_t^{+}(k_j)^{\top}\hat{v}_j
\end{aligned}\tag{34}
$$

所谓$q_t^{+}$仅仅是将$q_t$乘上了一个scale标量, 无伤大雅. 但是此时式子里有一个重要的东西:

$$
q_t^{+}(k_j)^{\top}\tag{35}
$$

这个东西是一个标量!

因此所有token:

$$
O=Q^{+}S+\operatorname{tril}[Q^{+}(K^{-})B_{\beta}]\hat{V}\tag{36}
$$

这样来看, 在KDA的运算里, 我们需要以下步骤:

<ol>
<li>通过 $(I+A)\hat{V}=V-K^{+}S$ 来计算出 $\hat{V}$</li>
<li>通过 $O=Q^{+}S+\operatorname{tril}[Q^{+}(K^{-})B_{\beta}]\hat{V}$ 计算出 $O$</li>
<li>通过 $\hat{s}_{out}=S+(K^{-})^{\top}B_{\beta}\hat{V}$ 得到下一个 $S$</li>
</ol>

讨论可行的Kernel实现思路:

我们还是先看看kimi官方怎么做的吧, 看上去把该kernel拆成了两个kernel.把chunk的大小固定为16了(难道不太小了吗?)

一个chunk开始的时候, 我们默认已经存在:

$$
S\in\mathbb{R}^{K\times V}\tag{37}
$$

官方实现里把chunk-local工作放给了K1, 把依赖上一个chunk的工作放到了K2.

K1的launch grid为$N\times H\times num\_chunks$

K2的launch grid为$N\times H$. 为了便于理解, 可以直接把H省略掉只剩下单头注意力.

回顾我们上文的推导, 原始的递推方程是这样的:

$$
S_i=D_iS_{i-1}+\beta_ik_i^{\top}(v_i-k_iD_iS_{i-1})\tag{38}
$$

我们定义了chunk内的decay:

$$
P_i=D_iD_{i-1}\ldots D_1=\operatorname{diag}(e^{g_i})\tag{39}
$$

然后进行了归一化:

$$
\hat{S}_i=P_i^{-1}S_i\tag{40}
$$

我们也已经推过:

$$
\hat{s}_t=\hat{s}_{t-1}+\beta_t(k_t^{-})^{\top}(v_t-k_t^{+}\hat{s}_{t-1})\tag{41}
$$

其中:

$$
k_i^{+}=k_i\cdot e^{g_i},k_i^{-}=k_i\cdot e^{-g_i}\tag{42}
$$

又定义:

$$
\hat{v}_i=v_i-k_i^{+}\hat{S}_{i-1}\tag{43}
$$

于是:

$$
\hat{S}_i=\hat{S}_{i-1}+\beta_i(k_i^{-})^{\top}\hat{v}_i\tag{44}
$$

这里我们相当于把之前的那些公式重新都写了一遍. 但是现在官方实现中引入了另一个变量把$\beta$吸收掉了:

$$
u_i=\beta_i\hat{v}_i\tag{45}
$$

那现在的state update就是:

$$
\hat{S}_i=\hat{S}_{i-1}+(k_i^{-})^{\top}u_i\tag{46}
$$

展开$\hat{S}_{i-1}$:

$$
\hat{S}_{i-1}=S+\sum_{j<i}(k_j^{-})^{\top}u_j\tag{47}
$$

注意到$P_0=I$, 所以$\hat{S}_0=S$

又有:

$$
\begin{aligned}
u_i&=\beta_i(v_i-k_i^{+}\hat{S}_{i-1})\\
&=\beta_i\left(v_i-k_i^{+}S-\sum_{j<i}k_i^{+}(k_j^{-})^{\top}u_j\right)
\end{aligned}\tag{48}
$$

展开:

$$
u_i+\sum_{j<i}\beta_ik_i^{+}(k_j^{-})^{\top}u_j=\beta_i(v_i-k_i^{+}S)\tag{49}
$$

嗯, 我们把问题实际放到kernel里, 实际上我们是想解这个problem.

定义:

$$
B_{\beta}=\operatorname{diag}(\beta_1,\ldots,\beta_C)\tag{50}
$$

以及:

$$
L=\operatorname{StrictTril}\left(B_{\beta}K^{+}(K^{-})^{\top}\right)\tag{51}
$$

则所有的token可以一次写作:

$$
U=(I+L)^{-1}B_{\beta}(V-K^{+}S)\tag{52}
$$

嗯可能我们看到这里已经有一些晕了, 但是重新想起来$U$相当于我们数学推导的时候的$\hat{V}$.

可以显然明显地看到, $L$的计算并不依赖$S$, 因此, 计算逆矩阵也并不依赖其他chunk的state, 但是其他的需要.

所以FlashKDA的大致分工即为:

## K1

K1负责计算T和S, 具体流程如下:

首先我们首先要构造的是一个chunk的

$$
G=\begin{bmatrix}
g_1\\
\vdots\\
g_C
\end{bmatrix}\tag{53}
$$

看上去是一个技术含量不高的scan.

之后构造:

$$
K^{+}=K\cdot e^G,K^{-}=K\cdot e^{-G},Q^{+}=Q\cdot e^G\tag{54}
$$

然后做第一个外积:

$$
K^{+}(K^{-})^{\top}\in\mathbb{R}^{C\times C}\tag{55}
$$

对于每一个元素乘上当前row的$\beta$, 然后取严格下三角:

$$
L=\operatorname{StrictTril}[B_{\beta}K^{+}(K^{-})^{\top}]\tag{56}
$$

然后计算逆矩阵$T$.

至于这个逆矩阵如何计算, 官方采用了纽曼级数的方法, 这也是我们选的C这么小的原因.

理论上K1的职责到此结束, 但是发现在$O$的计算中还是有一个可以并行计算的项:

$$
M_{qk}=\operatorname{Tril}(Q^{+}(K^{-})^{\top})\tag{57}
$$

好了, 如上可见, 每一个chunk可以正好并行.

## K2

K2的主要思路是串行进行所有chunk, 因为S具有依赖性(但是这非常不符合算子人的想法, 不知道对这个有无太多思路).

我们首先要计算:

$$
U=TB_{\beta}(V-K^{+}S)\tag{60}
$$

所以首先计算:

$$
K^{+}S:(16\times128)(128\times128)\rightarrow(16,128)\tag{61}
$$

然后V减去它是常见gemm操作.

之后乘$\beta$和K1算出的$T$:

$$
TR:(16\times16)(16\times128)\rightarrow(16\times128)\tag{62}
$$

然后我们便可以去计算O了:

$$
O=Q^{+}S+\operatorname{tril}[Q^{+}(K^{-})]U\tag{63}
$$

我们已经算过$M_{qk}$所以很容易地就可以算出$O$

剩下的最后一个问题是$S_C$

由之前的公式可以直接计算:

$$
S_C=P_CS+\sum_jP_C(k_j^{-})^{\top}U_j\tag{64}
$$

不妨直接设:

$$
k_j^{\mathrm{end}}=k_j\cdot e^{g_C-g_j}\tag{65}
$$

于是可以计算:

$$
S_{\mathrm{next}}=P_CS+(K^{\mathrm{end}})^{\top}U\tag{66}
$$

## epilogue

好了, 我们完成了对Kimi KDA attention的推导和对flash KDA v1 kernel的分析, 这是笔者第一次接触到linear attention kernel, 对于kda的一些设计也不能太过理解. 但是按照kimi k3的模型质量来看, 应该是没有训炸(x).

最近在对算子很感兴趣, 看flash kda v1的实现, 感觉并不算很成熟, 看flash infer里的一些pr也可以看出很容易地就可以实现2x的加速比.按照笔者的理解, 在K2里面只有head可以并行化, S串行化太过硬伤, 但是Chunksize又不能选大(选大之后neumann invmethod又会出现一些问题). 今天读了一下tian qi新发的论文CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution, 里面也提到了cake ir对kda的优化, 哎不知道对kda范式的革新是否还像fla一样由人类作出了...
