---
title: "Flash Attention CUDA Kernel 优化: 从 56 到 986 TFLOPS on B300 — G6: Cluster 与 2CTA 探索 (727→880)"
date: 2026-07-31 00:00:00
description: "A CUDA kernel optimization note for Flash Attention on B300. G6: cluster & 2CTA."
series: flash-attention-b300
series_order: 6
timeline_title: "G6 · Cluster 与 2CTA 探索"
timeline_metric: "727 → 880"
tags:
    - CUDA
    - kernel optimization
    - flash attention
categories:
    - practice
---
![cover](cover.png)

## 背景


G5 从 V21 到 V36，核心是 pipeline 重构 + 双缓冲 + 精细同步。V21 的 ping-pong 流水线把性能从 220 直接拉到 355 TFLOPS，后续经过一系列优化，最终收在 V35 的 727.25 TFLOPS。G5 的结论大概是：

1. pipeline 重构是收益最大的改动：让 TMA 和 tensor core 重叠，解决了之前串行等待的问题。
2. 指令副本膨胀需要控制：模板 × 循环展开的组合会导致 icache 压力，需要在外层做 unroll 1。
3. TC 利用率从 7.8% 涨到了约 18-20%：说明 pipeline 重构的方向是对的，但距离饱和还有距离。

G5 结束时隐约有一个感觉：单 CTA 的资源可能已经用得差不多了。无论怎么调 barrier、怎么压缩指令，TC 利用率就是上不去。也许问题不在 pipeline 本身，而在于一个 CTA 能调度的 warp 数量有限，无法让 tensor core 持续饱和。

于是 G6 打算尝试一个新方向：引入 cluster，用 2 CTA 来协同工作。


## V37: 2CTA 雏形 — 先跑起来再说

SM90 之后 NVIDIA 引入了 cluster 的概念，多个 CTA 可以协同工作，共享一些资源。之前一直只用 1 CTA，也许 occupancy 上不去的原因就在这里。

### 改动

为了写出正确的 2CTA kernel，回退了 G5 所有 barrier 上的细致优化。先把基础跑通，再考虑性能。

主要改动如下：

1. PV 改用 tcgen05 ts：QK 阶段得到的 P 存在 TMEM 里，之前 softmax 的时候会把 P 搬到 SMEM，现在让它一直留在 TMEM 里。

2. 2CTA 共享 KV 加载：两个 CTA 协同做 tcgen05，把 B 切片存在 SMEM 里，理论上能省一半的 TMA 流量。不过 softmax 这边不一定能直接受益，先跑起来看看。

3. KV 分成 6 个环（K、V 各 3 级交错）：前文省去了大量 SMEM 空间，现在可以容纳更多环。KV 的预加载粒度变细了，或许能增加 overlap 的机会。

还有一个值得关注的地方：w15 负责协调 cluster 内的指令。这个设计有点风险——w15 既要做 PV 发射又要做 cluster 同步，职责有点重。不过第一版先这样，后续再调整。

另外，SM103 大概有 148 个 SM，每个 CTA 占一个 SM 的话，2CTA 模式下 cluster 的分配可能会有一些不均匀。这会影响到负载均衡，但目前还只是小问题。
信号图大概如下：
![01](01.png)

### 性能

```text
Elapsed Time (total):      0.0051 seconds
Average Time Per Step:     0.051 ms
Compute Performance:       668.12 TFLOPS
Memory Throughput:         1215.31 GB/s
```

从 V35 的 0.047 ms / 727 TFLOPS 回退到了 0.051 ms / 668 TFLOPS。性能掉了大概 8%，但考虑到这是回退了所有 barrier 优化后的第一版，这个跌幅不算太夸张。

ncu 里 barrier stall 非常之多。说明当前的流水线还是依托答辩——2CTA 的同步开销比预期的要大，很多 warp 在等 barrier，计算单元没有充分利用起来。

不过至少跑通了，后续需要针对 cluster 场景重新做 barrier 的精细优化，把 G5 积累的经验移植过来。

---

## V38: setmaxnreg — 运行时寄存器限制

G5 以来，spill 主要集中在 softmax 相关的代码段。分析发现，wg0/wg1 在 softmax 阶段寄存器需求较高（约 176 个），而 wg2/wg3 需求较低（不到 80 个）。当前 `__launch_bounds__(512, 1)` 统一按 128 寄存器编译，导致 softmax 区域生成 spill。

### 改动

PTX 提供了 `setmaxnreg` 指令，可以在运行时调整 warp 的寄存器上限：

```cpp
template<uint32_t RegCount>
__device__ __forceinline__ void setmaxnreg_inc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(RegCount));
}

template<uint32_t RegCount>
__device__ __forceinline__ void setmaxnreg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(RegCount));
}
```

在 kernel 入口处对不同 warp group 做差异化设置：wg0/wg1 升至 176 寄存器，wg2/wg3 降至 80 寄存器，期望通过运行时按需分配来消除 softmax 区域的 spill。

### 性能

```text
Elapsed Time (total):      0.0051 seconds
Average Time Per Step:     0.051 ms
Compute Performance:       669.47 TFLOPS
Memory Throughput:         1217.76 GB/s
```

相比 V37 的 668 TFLOPS，无明显变化。ncu 显示寄存器分配仍为 128，spill 没有减少。

### 原因

`setmaxnreg` 只控制运行时寄存器分配上限，不能撤销编译器已经生成的 spill。ptxas 在编译时按 `__launch_bounds__(512, 1)` 的 128 寄存器目标完成了寄存器分配和 spill 决策，local memory 的 load/store 已经写入指令流。运行时动态调整上限不会让编译器重新决策。

此方向不可行，转向流水线排布优化。
---
## V39: Interleave — 交错 QK/PV 发射

参考了 Flash Attention 4 的实现，对每个 phase 的 mma 发射顺序做了调整。

### 改动

原来的顺序是按 key 分组，每个 key 完整做完 QK 和 PV 后再进入下一个 key：

```
for key = 0..7:
    等 kv_full[K] → 发 QK0(key), QK1(key)
    等 p_cluster[key][0/1] → 发 PV0(key), PV1(key)
```

问题在于：QK(key+1) 必须等当前 key 的 PV 全部完成才能开始，导致 softmax → PV → QK 形成串行链。

新顺序改为交错发射：

```
开始: 发 QK0(0), QK1(0)

稳态 (key = 0..6):
    等 p_cluster[key][0] → 发 PV0(key)
    等 kv_full[key+1] → 发 QK0(key+1)
    等 p_cluster[key][1] → 发 PV1(key)
    发 QK1(key+1)

收尾: 发 PV0(7), PV1(7)
```

核心变化是 QK 不再被上一阶段的 PV 阻塞，而是整体提前了一个 phase。由于当前采用 6 环 KV 流水线（K、V 各 3 级交错），提前发射 QK 不会造成数据冒险。

### 性能

```text
Elapsed Time (total):      0.0047 seconds
Average Time Per Step:     0.047 ms
Compute Performance:       726.84 TFLOPS
Memory Throughput:         1322.12 GB/s
```

回到 G5 SOTA 水平。ncu 数据：barrier 2.88（中继 + per-task 重 init）、long_sb 5.99（softmax 的 TMEM 物理往返）、spill 476k。
---
## V40: Barrier PTX — 拒绝外联

CUTLASS 的 Barrier::wait() 编译后，ptxas 会把自旋重试路径（PHASECHK）外联成慢路径代码。更糟的是，barrier 地址是运行期算的（key 是循环变量），这个地址寄存器要跨越外联边界存活——ptxas 的处理是把它 spill 到 local memory。

结果就是 softmax warp 每次在 score_ready 上醒来重试，都要先从 local memory 把 barrier 地址读回来。spill 走到了关键路径上。

既然这样，那我们不用你的。直接手写一个 barrier wait：

```cpp
asm volatile(
    "{\n\t"
    ".reg .pred score_ready;\n\t"
    ".reg .b32 score_addr;\n\t"
    "mad.lo.u32 score_addr, %0, 16, %1;\n\t"
    "SCORE_WAIT:\n\t"
    "mbarrier.try_wait.parity.shared::cta.b64 "
    "score_ready, [score_addr], %2, %3;\n\t"
    "@!score_ready bra SCORE_WAIT;\n\t"
    "}\n"
    : : "r"(static_cast<uint32_t>(key)), "r"(base), "r"(phase), "r"(ticks)
    : "memory"
);
```

拒绝外联!

```text
Elapsed Time (total):      0.0047 seconds
Average Time Per Step:     0.047 ms
Compute Performance:       727.64 TFLOPS
Memory Throughput:         1323.58 GB/s
```

性能没涨。但 spill 从 47 万降到了 7 万。至少关键路径上那点 spill 污染被清掉了。

## V41: Conditional Scale — 允许 b 滞后

我们之前的 softmax 求出最大值之后会直接用这个 max 去做 exp。但仔细想想，我们真的需要严格的无溢出吗？max 存在的意义是防止指数运算溢出。如果把数值控制在合理范围内，那 b 也是可以有一些宽松处理的。

基于这个思想，我们做了一个 conditional scale（换基判定）：

流程大概是这样的：

① QK 完成，S 是 128 个原始点积（fp32，还没乘 scale）  
② `tile_max` = 128 个 s 的最大值（4 次 ld.red）  
③ `candidate = max(旧b, tile_max × ScaleLog2)`  
④ 换基判定：  
&nbsp;&nbsp;— `candidate ≤ 旧b + 8` ? 是 → b 不动，history = 1（绝大多数情况）  
&nbsp;&nbsp;— 否 → b = candidate，history = 2^(旧b - 新b)（只在 max 快速上升时触发）  
⑤ 逐 score：`p = 2^(s·ScaleLog2 - b)`，转 fp16，打包写回 TMEM（P区）  
&nbsp;&nbsp;— 因为 b 允许滞后 8，p 的范围变成 (0, 256]，不再是 (0, 1]  
⑥ tile 内求和：`packed_sum` 用 half2 累加 → `tile_sum(fp32)`  
⑦ `row_sum = row_sum × history + tile_sum`，history=1 时原样保留  
⑧ `history_scale[row] = history`，交给 wg2

然后 rescale 阶段：

```cpp
fp32_t scale = storage.history_scale[row];  // ① 从 smem 读自己的系数

if (__ballot_sync(0xffffffff, scale < 1.0f) == 0u)  // ② warp 投票
    return;  // 32 行全是 1.0 → 整个 warp 直接走人

uint32_t output_base = tmem_base + (warp&3)*32<<16 + Output区;  // ③ 自己 warp 的 32 行

for (vector = 0..7):  // ④ 128 列分 8 趟
    load_32x16(output_base + vector*16, values)  // TMEM 读 16 列 fp32
    wait_load()
    for (i = 0..15):
        values[i] *= scale  // 16 条 FMUL
    store_32x16(...)  // 写回 TMEM

fence_before_thread_sync()  // ⑤ 保证写回对后续 MMA 可见
```

因为整个 warp 里的 32 行用的是同一个 scale，`__ballot_sync` 一票否决，要么全做要么全不做。所以 warp divergence 是不会有的。

这真的是穷途末路，无计可施了。硬凑出来的。

```text
Elapsed Time (total):      0.0043 seconds
Average Time Per Step:     0.043 ms
Compute Performance:       795.37 TFLOPS
Memory Throughput:         1446.77 GB/s
```

效果还挺不错。指令数降了 17%，tensor core 利用率到了 31%。可以认为这一版提升是有效的。
---
## V42: Barrier 合并

p_cluster 和 correction cluster 这两组 barrier 可以合并，省一次同步。

```text
Elapsed Time (total):      0.0043 seconds
Average Time Per Step:     0.043 ms
Compute Performance:       796.63 TFLOPS
Memory Throughput:         1449.06 GB/s
```

收益不大，可改可不改。


## V43: PV 细分

尝试把 P 拆成前 96 列 + 后 32 列。softmax_tile 把 max 计算和 history 计算提到 exp 循环之前，算完立刻 arrive。wg2 的 rescale 可以和 softmax 的 exp 循环并行——原来 wg2 要等整个 softmax 结束。

依赖链的变化：

旧：`softmax(全128列) → PV(8块)`  
新：`softmax(96列) → PV(6块)` + `softmax(余32列) → PV(2块)`，PV 起点从 "softmax 100% 完成" 提前到 "75% 完成"，TC 提前介入；rescale 也从 softmax 尾部移到头部之后。粗粒度的生产者-消费者关系被细化成流水。

```text
Elapsed Time (total):      0.0041 seconds
Average Time Per Step:     0.041 ms
Compute Performance:       837.10 TFLOPS
Memory Throughput:         1522.67 GB/s
```

long scoreboard 下降了 18%，这是真的精修。


## V44: Remove Rescale WG

尝试把 rescale wg 移除。现在 softmax 的逻辑是：

```cpp
// softmax_tile 里，算完 history 之后：
if (key != 0) {
    bool needs_rescale = __ballot_sync(全warp, history < 1.0f) != 0u;
    if (needs_rescale) { // 稀有路径
        pv_done[key-1][Stage].wait(0); // 只有真要改 O 才等上一笔 PV
        rescale_output<Stage>(storage);
    }
    correction_done[key][Stage].arrive(); // 常态：立刻放行
}
```

epilogue 也交给 wg0/wg1 来做。现在变成 3 个 wg，总共 384 线程，寄存器变多了。

```text
Elapsed Time (total):      0.0039 seconds
Average Time Per Step:     0.039 ms
Compute Performance:       880.14 TFLOPS
Memory Throughput:         1600.96 GB/s
```

spill 消失了，168 reg per thread。long scoreboard 也少了 33%，推测是因为不需要等到 rescale 了。

后面尝试了多版本优化，都没能突破 0.039 ms。用 ncu 逆向看了一下 torch sdpa，发现它用的是 1 CTA。于是我们决定换技术路线，回到 1 CTA 继续优化。

---

## G6 小结

G6 从 V37 到 V44，核心主题是 2CTA cluster。

V37 首次引入 2CTA，性能从 G5 的 727 掉到 668。V38 的 setmaxnreg 尝试失败，V39 的 interleave 回到 726，V40 的手写 barrier PTX 把 spill 从 47 万降到 7 万，V41 的 conditional scale 直接冲到 795，V42 合并 barrier 微涨到 796，V43 的 PV 细分到 837，V44 移除 rescale wg 后最终收在 880.14 TFLOPS。

### 性能演进

| 版本 | 核心改动 | Time/step | TFLOPS |
|------|---------|-----------|--------|
| V35（G5 终点） | rescale ld/store 位宽 16 | 0.047 ms | 727.25 |
| V37 | 2CTA 雏形 | 0.051 ms | 668.12 |
| V38 | setmaxnreg | 0.051 ms | 669.47 |
| V39 | Interleave QK/PV | 0.047 ms | 726.84 |
| V40 | 手写 barrier PTX | 0.047 ms | 727.64 |
| V41 | Conditional scale | 0.043 ms | 795.37 |
| V42 | 合并 barrier | 0.043 ms | 796.63 |
| V43 | PV 细分 96+32 | 0.041 ms | 837.10 |
| V44 | 移除 rescale wg | 0.039 ms | 880.14 |

G6 从 668 爬到 880，整体提升了约 32%。其中 V41 的 conditional scale 是单版收益最大的改动，从 727 到 795，涨了约 9%。

### 几个比较明确的结论

1. 2CTA 本身不直接带来收益：V37 比 G5 终点还低。cluster 的价值在于提供了更多的寄存器资源和并发度，但需要配合其他优化才能释放出来。

2. b 允许滞后 8 是个好思路：大多数 tile 的 max 不会剧烈变化，history=1 的常态路径 bypass 了 rescale，省掉了大量指令。只有在 max 快速上升时才触发换基，这个设计把“保险”和“常态”分开了，是非常实用的工程手法。

3. PV 细分把生产者-消费者关系打得更散：softmax 不需要全部做完才能开始 PV，96 列够用就先发一批，TC 提前介入。这种“边生产边消费”的思路比一次性算完再交棒更高效。

4. torch sdpa 用的是 1CTA：ncu 逆向确认了这一点，说明 1CTA 在 SM103 上并没有资源受限的问题。既然如此，搞 2CTA 反而多了一套 cluster 同步的开销，性价比存疑。

### 下一步方向

既然 torch sdpa 用 1CTA 能跑到 1156 TFLOPS，说明 1CTA 的天花板还没摸到。G6 在 2CTA 上验证的技术（conditional scale、PV 细分、手写 barrier）可以迁移回 1CTA，看看能否在更简单的同步模型下继续往上优化。
