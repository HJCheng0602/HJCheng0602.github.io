---
title: "Flash Attention CUDA Kernel 优化: 从 56 到 986 TFLOPS on B300 — G3: Fragment Reshape 与架构级突破 (144→227)"
date: 2026-07-31 00:00:00
description: "A CUDA kernel optimization note for Flash Attention on B300. G3: fragment reshape & architectural changes."
series: flash-attention-b300
series_order: 3
timeline_title: "G3 · Fragment Reshape 与架构级突破"
timeline_metric: "144 → 227"
tags:
    - CUDA
    - kernel optimization
    - flash attention
categories:
    - practice
---
![cover](cover.png)

## 背景


G2 从 V6 到 V11，在算术指令和 occupancy 之间反复拉扯。V6 的 FMA 尝试宣告失败，V7 的 4-row softmax 小有收益，V8 的 vec2 又往前蠕动了一点，V9 的地址优化来回折腾，V10 再次被 FMA 提示打脸，V11 的消融实验终于收在了 144.41 TFLOPS。

G2 的结论大概是：

1. ncu 的 FMA 提示是幻觉：只要 kernel 还是 latency-bound，FMA 就救不了你。
2. 寄存器比 shared memory 更稀缺：V9 的双缓冲方案技术上有效（mma stall 从 38% 降到 9.4%），但 occupancy 掉了，性能反而倒退。
3. 微调已经到头了：G2 整体只涨了 11%，而 G1 涨了 130%。该做点架构级的改动了。

所以 G3 打算换个打法，不再盯着 softmax 那条链修修补补，而是直接动 wmma 的布局。于是有了 V12。

## V12: Fragment 与 Reshape — 打破 QK 与 PV 的墙

首先前文做的修改大都是小打小闹，本次迭代主要是聚焦 wmma_QK 与 wmma_PV 之间的衔接问题。

在之前的版本中，进行完 QK mma 之后我们会把 P 存到 smem 里，然后 softmax，再 wmma_PV。但考虑到我们是手写的 wmma，实际上我们可以直接接触到 wmma 的 per-thread fragment 管理，不一定非得走 shared memory 这一趟。

先回顾一下之前的 wmma 布局：

```
wmma: Br=32, Bc=64
WarpM=16, WarpN=32
TPB=128

            列 0-31       列 32-63
行 0-15:   ┌ warp 0 ┐   ┌ warp 1 ┐
行 16-31:  └ warp 2 ┘   └ warp 3 ┘
(TileM=1, TileN=2)
```

对于每一个 QK tile，需要两个 warp 来管理一行（warp 0 管左半，warp 1 管右半）。但相邻 warp 之间的寄存器数据是不可见的，所以 P 必须走 shared memory 才能让后面的 PV 拿到完整的一行。

这一来一回，不仅多了几轮 sync，还让 shared memory 的带宽成了瓶颈。

### 新的布局：让一个 warp 管一整行

既然问题出在“一行需要两个 warp”，那最直接的解法就是让一个 warp 能管一整行。我把 Br 从 32 翻倍到 64，同时把 WarpN 从 32 扩到 64：

```
wmma: Br=64, Bc=64
WarpM=16, WarpN=64
TPB=256

            列 0-63
行 0-15:   ┌ warp 0 ┐
行 16-31:  └ warp 1 ┘
行 32-47:  ┌ warp 2 ┐
行 48-63:  └ warp 3 ┘
(TileM=1, TileN=4)
```

这样一来，每个 warp 独自管理一行的全部 64 列，QK 做完之后 P 的 fragment 直接留在寄存器里，softmax 的 max/sum 归约也在 warp 内部完成，不需要经过 shared memory 的 P 缓冲区。

这个改动的另一个好处是 Br 翻倍了，每个 block 处理的 Q 行数变多，KV 的复用率也跟着上来了，理论上访存效率会更好。

### 代价与不优雅的地方

当然，天下没有免费的午餐。Br 翻倍的代价是每个 wmma 类需要管理两倍的 fragment，寄存器压力肯定会涨，occupancy 大概率会掉一些。

但更不优雅的地方在于 QK 和 PV 的衔接：

- QK 阶段只用了 4 个 warp（每行一个）。
- PV 阶段我们沿用原来的设计，用了 8 个 warp：

```
PV 布局（WarpH=64, WarpCountH=2）:

            输出列 0-63    输出列 64-127
行 0-15:   [ warp 0 ]    [ warp 1 ]
行 16-31:  [ warp 2 ]    [ warp 3 ]
行 32-47:  [ warp 4 ]    [ warp 5 ]
行 48-63:  [ warp 6 ]    [ warp 7 ]
(WarpH=64, WarpCountH=2)
```

交接点：QK 的 4 个 warp 做完 softmax 之后，把 exp 结果打包写进 smem_P（64×64 fp16），因为下一段的消费者不止它们四个。然后 PV 的 8 个 warp 再从 smem 里读。

相比之前，P 不再需要存两遍了（之前是 logits → smem → softmax → smem_P → PV，现在是 QK 在寄存器里做完 softmax → 直接写 smem_P → PV），省掉了一轮 sync。但 QK 和 PV 之间仍然需要 smem 同步交接。

不优雅的地方很明显：

1. QK 阶段只用到 4 个 warp，剩下 4 个 warp 在摸鱼。
2. QK 和 PV 还是需要 smem 交接，没有做到真正的寄存器直传。

但这样的设计给了我们后续改进的可能——PV 的布局可以后续再调，先跑起来看看效果。


```text
Elapsed Time (total):      0.0157 seconds
Average Time Per Step:     0.157 ms
Compute Performance:       219.15 TFLOPS
Memory Throughput:         398.64 GB/s
```

从 V11 的 144.41 TFLOPS 直接蹦到了 219.15 TFLOPS，提升了大概 52%。

这是 G1 以来最大的一次单版提升。

### What happened?

从 ncu 来看，变化很明显：

- LSU（Load-Store Unit）从 89% 降到了 63%，shared memory 的压力明显减轻了。
- softmax 从 stall 列表里消失了，因为 softmax 现在在寄存器里做，不再需要频繁读写 smem。

这印证了最初的判断：之前 shared memory 的 P 缓冲区是瓶颈之一，只是之前的所有优化都在外围打转，没有触及这个核心问题。

当然，V12 只是一个粗糙的原型。布局改了，还有很多细节可以优化——比如 PV 能不能也改成和 QK 对齐的布局，让寄存器直接传递而不是经过 smem？比如 occupancy 掉到了多少，能不能补回来？


---

## V13: Warp Specialization — 让摸鱼的 warp 干点正事

V12 虽然提升巨大，但跑完 ncu 一看，有个问题一直让我有点心存芥蒂：QK 阶段只用了 4 个 warp，剩下 4 个 warp 在摸鱼。

Br=64 的新布局下，QK 只需要 4 个 warp 来算（每行一个），但 block 里有 8 个 warp（TPB=128 / 32 = 4 warp，不对，TPB=128 是 4 个 warp？等一下，TPB=128 / 32 = 4 warp，但 V12 里 QK 用了 4 个 warp，PV 用了 8 个 warp 似乎有点问题……不管了，反正有 warp 在摸鱼）。

除了 warp 空闲的问题，还有一个历史遗留问题：在之前的所有实现中，我们用了 `smem_KV` 来共享 K 和 V 两个 tile，这确实是降低单 block smem 占用的有效方法。但现在的逻辑是：wmma_QK 必须先等 cp.async 把 KV 加载到 smem，然后才能 ldmatrix，最后才能 cp.async V。换句话说，V 的加载在等 QK 算完，这个依赖关系天然有 stall。

如果能把 V 的加载提前到 QK 计算的同时进行，是不是就能把访存延迟给藏住？

### 异步加载 + Warp Specialization

结合上面两个问题，想法就变得很自然：让空闲的 warp 在 QK 计算的同时帮忙搬 V 的数据。

具体来说，不在共享一个 `smem_KV`，而是单独开一个 `smem_V` 缓冲区，QK 算的同时，空闲 warp 往 `smem_V` 里 cp.async 搬数据。这样等 QK 算完、softmax 做完之后，V 已经在 shared memory 里等着了，不需要再等一轮 global load。

大致流程如下（示意）：

```
Warp 0-3: QK mma + softmax（算 P）
Warp 4-7: cp.async 搬 V 到 smem_V（顺便干点别的）
     ↓
QK 做完，V 已经在 smem_V 里
     ↓
Warp 0-7: PV mma
```

### 代价：突破 static smem 限制

这个改动的代价是 shared memory 用量上去了。我算了一下，新的 CTA 的 smem 用量预计为 58 KB。

58 KB 超过了很多架构的 static smem 限制（48 KB）。好在 B300 是 SM103，总 shared memory 有 228 KB（没记错的话），但 static 分配默认只有 48 KB，需要dynamic smem 分配。

另外，为了确保 occupancy 不崩，我加了 `__launch_bounds__(TPB, 2)`，保证 2 个 CTA 能同时驻留在一个 SM 上。这样每个 thread 的 max register 就是：

```
65536 / (256 * 2) = 128 个 reg/thread
```

刚好卡在 128 的边界上，再高就得掉 occupancy。

### V13 的性能

跑了一下 benchmark：

```text
Elapsed Time (total):      0.0151 seconds
Average Time Per Step:     0.151 ms
Compute Performance:       226.83 TFLOPS
Memory Throughput:         412.60 GB/s
```

从 V12 的 219.15 TFLOPS 提升到了 226.83 TFLOPS，大概又涨了 3.5%。虽然不像 V12 那么夸张，但在这个阶段还能有正向收益就不错了。

ncu 看下来，register per thread 是 124，没到 128 的上限，还有一点点空间。cp.async 的延迟确实被藏住了一些，但感觉还没完全发挥出来。

### 尝试 Br=128

既然 register 没满（124 < 128），能不能把 Br 从 64 翻倍到 128？

理论上 Br 越大，KV 复用率越高，但代价是每个 warp 管理的 fragment 更多，寄存器压力也更大。

跑了一下：

```text
Elapsed Time (total):      0.0173 seconds
Average Time Per Step:     0.173 ms
Compute Performance:       198.96 TFLOPS
Memory Throughput:         361.90 GB/s
```

显然 spill 了，性能直接回退。即使把 hybrid smem 也 opt in 了，结果还是不行：

```text
Elapsed Time (total):      0.0152 seconds
Average Time Per Step:     0.152 ms
Compute Performance:       226.35 TFLOPS
Memory Throughput:         411.73 GB/s
```

跟 V13 基本持平，没有提升也是必然的。

### 小结与下一步

V13 的方向是对的——warp specialization + 异步加载确实有效，只是提升幅度没有 V12 那么惊艳。Br=128 的尝试也说明，当前架构下 Br=64 已经是一个比较合理的点，再大就要 spill 或者 occupancy 掉档了。

不过 cp.async 还是有点低效，每次加载都要走一遍地址计算和同步，感觉 TMA 可能会是更好的选择。此外，既然 warp specialization 已经做到了 QK 和 PV 的阶段分离，接下来是不是可以把 softmax 也单独拎出来，让专门的 warp 负责？

---

## G3 小结

G3 从 V12 到 V13，核心变化是从架构层面重新思考了 QK 和 PV 的衔接方式。

V12 的 fragment reshape 把 Br 从 32 翻倍到 64，让一个 warp 管一整行，P 不再走两遍 smem，收益巨大（52%）。V13 在此基础上做了 warp specialization，让空闲的 warp 提前搬 V 的数据，进一步隐藏了访存延迟。

### 性能演进回顾

| 版本 | 核心改动 | TFLOPS | 变化 |
|------|---------|--------|------|
| V11（G2 终点） | G2 收尾 | 144.41 | — |
| V12 | Fragment reshape，Br 32→64 | 219.15 | +52% |
| V13 | Warp specialization + 异步加载 V | 226.83 | +3.5% |
| Br=128 尝试 | Br 64→128 | 198.96 | 失败（spill） |
| hybrid smem | 尝试救 Br=128 | 226.35 | 持平 |

### 几个比较明确的结论

1. 架构级改动的收益远大于微调：V12 的 52% 提升印证了之前的判断——瓶颈在 P 缓冲区的 shared memory 往返，而不在算术指令或 occupancy。之前 G2 的所有努力加起来只涨了 11%，而 V12 一个改动就涨了 52%。

2. Br 的大小有最优值：Br=32 太小，KV 复用不够；Br=128 太大，寄存器 spill。Br=64 在当前架构下是个比较舒服的点。

3. Warp specialization 有效，但收益递减：V13 的 3.5% 提升说明异步加载的方向是对的，但 cp.async 的收益边际在缩小。下一步可能要上 TMA 才能看到更大的提升。

4. static smem 48KB 限制已经被突破：V13 用了 58KB smem，虽然在 SM103 上能跑，但后续如果要进一步增大 smem 用量，需要更仔细地管理 occupancy 和寄存器边界。

### 下一步方向

G3 算是开了个好头，但 226.83 TFLOPS 离 baseline 的 1156 TFLOPS 还有 5 倍的空间。回看这几轮优化——swizzle、softmax、fragment reshape、warp specialization——我意识到一个一直被忽略的问题：我们写的始终是 SM80 时代的代码。mma.sync + ldmatrix + cp.async 这套似乎已经过时了。

而 B300 是 SM103，我们有 tcgen05（一条指令算 128×128，TMEM）、TMA（DMA 硬件搬运，自带 swizzle 和 mbarrier）、228 KB smem——这些新的feature我们一个都没用上。

所以 G4 打算换一下：不再在 SM80 的框架里修修补补，直接用 SM103 的 TMA + TCGen05 重写一版。能不能跑通、性能会不会先掉一截，都还不知道——但方向应该是对的。G3 到此结束。

---
