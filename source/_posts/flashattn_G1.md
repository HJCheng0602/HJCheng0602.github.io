---
title: "Flash Attention CUDA Kernel 优化: 从 56 到 986 TFLOPS on B300 — G1: 一切的开始 (56→130)"
date: 2026-07-31 00:00:00
description: "A CUDA kernel optimization note for Flash Attention on B300."
series: flash-attention-b300
series_order: 1
timeline_title: "G1 · 一切的开始"
timeline_metric: "56 → 130"
tags:
    - CUDA
    - kernel optimization
    - flash attention
categories:
    - practice
---

![cover](cover.png)

## 背景

本周学习目标是对 Flash Attention 做 CUDA kernel 优化，跑在 B300 上。我们的 case 大概是这样的：
```
SEQ_LEN = 1024
NUM_HEADS = 64
HEAD_SIZE = 128
DTYPE = torch.float16
```

先跑了一下 PyTorch 原生的 `torch.nn.functional.scaled_dot_product_attention` 作为 baseline，数据大概是：

```text
Function: torch_sdpa[1024, 64, 128]
Elapsed Time (total):      0.0030 seconds
Average Time Per Step:     0.030 ms
Compute Performance:       1156.54 TFLOPS
Memory Throughput:         2103.73 GB/s
```
嗯，baseline 已经很快了。接下来要做的就是从一个 naive 的实现慢慢优化，而且用ncu看了一下细节，这个算子走的是cudnn auto 调度路径，ncu 逆向确认是 1 CTA。

## V1: 一切的开始
第一版 kernel 的逻辑比较直接，主要参考了 Flash Attention 官方实现的结构，然后做了些适配。

### 启动参数
```cpp
TPB = 128
Br  = 32
Bc  = 64

WarpM = 16
WarpN = 32
WarpH = 64

Grid = (SEQ_LEN / Br, NUM_HEADS)  // 即 (32, 64)
```
也就是说，每个 block 处理一个 head 中的一个长度为 Br=32 的 Q 片段。总共 32 × 64 = 2048 个 block，每个 block 128 个 thread，感觉规模还行。

### 共享内存布局
算了一下需要的 shared memory 大小，大概是这样：

```cpp
fp16_t smem_Q[Br * HeadSize];       // 32 * 128 = 4096 个 fp16 = 8 KB
fp16_t smem_KV[Bc * HeadSize];      // 64 * 128 = 8192 个 fp16 = 16 KB
fp16_t smem_P[Br * Bc];             // 32 * 64  = 2048 个 fp16 = 4 KB
fp32_t smem_Logits[Br * Bc];        // 2048 个 fp32 = 8 KB
fp32_t smem_max[Br];                // 32 个 fp32 = 128 B
fp32_t smem_max_old[Br];            // 32 个 fp32 = 128 B
fp32_t smem_sum_exp[Br];            // 32 个 fp32 = 128 B
```
加起来大概 36 KB 左右，B300 的 单block shared memory 默认是 48 KB（除非optin），所以应该放得下，至少不会一开始就爆 shmem。

### Pipeline 思路
整个 pipeline 大概是这样：

1. 初始化 smem_max 和 smem_sum_exp。

2. 用 ldmatrix 加载 Q 到 寄存器Q。

3. 然后进入 KV 的大循环，每次取 Bc=64 长度的 KV 片段。

4. 对每个 KV 片段：

5. 加载 K，做 QK matmul，得到 logits 存到 smem_Logits。

6. 做 online softmax，更新 smem_max 和 smem_sum_exp。

7. 加载 V，做 PV matmul，累加到 O 上。

8. 循环结束后，把 O 写回 global memory。

看起来逻辑是通的，至少没有明显的 bug。

### 关于 Swizzle 的一点琢磨
QK 的 mma 我用了 Swizzle64，KV 的 mma 用了 SwizzleNone。当时选择的时候大概是这么想的：

QK 这边，ldmatrix 加载 A/B 矩阵，row stride = HeadSize * 2 = 256B。因为选的是 x4 模式，每个 lane 提供一个 16B 行的地址，加载 4 个 8×8 tile。shared memory 一个周期的带宽是 128B，一个 tile 恰好占一个周期。

如果 tile 落在 lane 0–7, rows 0–7, col c 的位置，用 Swizzle64 会产生 2-way bank conflict。我算了一下 QK 的指令数：TileM=1, TileN=2，总共 24 条 ldmatrixx4，mma 指令 32 条。我猜测 ldmatrix 大概能被 mma 掩盖掉，所以 Swizzle64 的 bank conflict 或许不是主要瓶颈。当然这只是我的推测，后面看 profiling 才知道。

PV 那边用了 SwizzleNone，因为 WarpH=64, Bc=64，感觉不太需要额外的 swizzle 来优化 bank 分布。

### V1 拉完了
满怀期待地跑了一下 benchmark，结果：

```text
Elapsed Time (total):      0.0605 seconds
Average Time Per Step:     0.605 ms
Compute Performance:       56.83 TFLOPS
Memory Throughput:         103.38 GB/s
```

跟 baseline 的 1156 TFLOPS 相比，差了大概 20 倍。但是第一版嘛，能跑对就不错了，性能的事后面慢慢搞。

### 初步猜测
从 profiling 数据来看，有几个指标比较突出，我大概觉得瓶颈可能在以下几个地方：

- Shared memory bank conflict 偏高：尤其是 smem_Logits 的读写，或者 online softmax 那一段的访问模式可能不太规则，导致了较多的 shared memory replay。

- Warp stall 占比比较大：可能是因为 wait_group 的等待时间太长，或者 mma 指令之间没有足够的独立指令来 hiding latency，导致 warp 经常在等数据。

- Occupancy 或许不够理想：虽然 shmem 只用了 36KB，但 TPB=128 加上寄存器压力，可能 active warp 数并没有达到预期，SM 的并行度没有完全发挥出来。


V1 到这里就先告一段落，下一步应该是先定位最明显的 bottleneck，然后有针对性地改。目前的想法是，先从 shared memory 的访问模式入手，看看能不能优化一下 bank conflict，然后再考虑 warp-level 的调度问题。

---

## V2: PV Swizzle 优化

在 V1 的 profiling 数据里，我注意到一个之前可能被我忽略的问题：PV 的 swizzle 还是需要设的。

V1 里 PV 的 mma 我用了 `SwizzleNone`，当时想着 `WarpH=64, Bc=64`，感觉不太需要额外处理。但仔细一想，`SwizzleNone` 在这种场景下可能反而会引入比较严重的 bank conflict。

### 溯因

PV 这边，P 的 shape 是 `[Br, Bc] = [32, 64]`，存的是 fp16，所以 row stride = `64 * 2 = 128B`。

用 `ldmatrix` 从 shared memory 加载 P 时，如果采用 `SwizzleNone`，一列数据会落到固定的 bank 上。128B 的 row stride 意味着相邻行之间的地址偏移是 128B，而 shared memory 有 32 个 bank，每个 bank 4B，总带宽 128B/cycle。每一行的同一列会命中完全相同的 bank，这就造成了 8-way bank conflict（128B / 16B = 8，因为每个 bank 在同一个 cycle 只能服务一个请求）。

同理，V 的加载也存在类似的问题。

我从指令数和 tensor core 占用的角度大概算了一下：

| 阶段 | TileM × TileN × TileK | ldmatrix.x4 条数 | 硬件 mma 条数 | Tensor Core 占用 |
|------|----------------------|-----------------|--------------|------------------|
| QK forward | 1 × 2 × 8 | 8 × (1+2) = 24 | 32 | ~256 cycles |
| PV forward | 1 × 4 × 4 | 4 × (1+4) = 20 | 32 | ~256 cycles |

从表上看，QK 和 PV 的 mma 指令数都是 32 条，tensor core 占用都在 256 cycles 左右，理论上两者的计算密度应该是接近的。但 PV 这边因为 `SwizzleNone` 导致的 8-way bank conflict，`ldmatrix` 的有效延迟会显著增加——8-way conflict 大概会导致 640 cycles 的 load 延迟，这个数字已经超过了 mma 本身的 256 cycles。

换句话说，在 V1 里，PV 的 shared memory 访问延迟已经成了主导瓶颈，计算单元在等数据。

### 改用 k128B

既然问题是 row stride = 128B 导致的 bank conflict，那最直接的思路就是改变 swizzle 模式，让 bank 分布更均匀。我决定把 PV 的 swizzle 从 `SwizzleNone` 改成 `Swizzle128`。

k128B 意味着把 128B 的 row stride 打散，使得同一列的相邻行映射到不同的 bank 上，从而减少 conflict。代价是 shared memory 的地址计算会稍微复杂一点，但这个 tradeoff 在当前的瓶颈面前应该可以忽略。

另外，V 这边用的是 `ldmatrix.trans`，本身就比普通的 `ldmatrix` 更难做到 1 cycle 一次，bank conflict 会更敏感。改完 swizzle 之后，理论上对 V 的加载也会有改善。

顺带一提，online softmax 那一段的存储布局也跟着做了些调整，不过那部分改动比较细节而且也是跟着swizzle的布局修正，就不展开了。

### V2 的性能

改完 PV 的 swizzle 之后重新跑了一下 benchmark：

```text
Elapsed Time (total):      0.0376 seconds
Average Time Per Step:     0.376 ms
Compute Performance:       91.46 TFLOPS
Memory Throughput:         166.37 GB/s
```

相比 V1 的 56.83 TFLOPS，提升到了 91.46 TFLOPS，大概提升了 1.6 倍。虽然跟 baseline 的 1156 TFLOPS 还有很大差距，但至少方向是对的。

从 profiling 数据来看，shared memory bank conflict 确实降了不少，说明改 swizzle 这个方向没问题。不过目前还远没有到可以庆祝的时候，91 TFLOPS 离目标还有至少一个数量级的距离。

下一步大概要看看其他瓶颈了，通过ncu看到 warp stall 和 occupancy 可能还是有问题，需要再仔细分析一下。

---

## V3: 进一步 Swizzle 优化

前文之述备矣，QK 这边其实也有类似的 bank conflict 问题，只不过没有 PV 那边那么严重。

### 再看 QK 的 Bank Conflict

QK 这边 row stride = `HeadSize * 2 = 256B`，shared memory 一个 section 是 16B（4 个 bank × 4B），256B / 16B = 16 个 section。如果用 `Swizzle64`，会把 64B 的跨度映射到相同的 bank 组上，导致 2-way bank conflict。

我之前算过，QK 的 `ldmatrix` 有 24 条，加上 mma 有 32 条，觉得 2-way conflict 大概能被 mma 掩盖掉。但是从 profiling 数据来看，虽然 QK 的冲突确实没有 PV 那么夸张，但它还是存在的，而且日积月累下来也会对性能有影响。

既然 V2 把 PV 从 `SwizzleNone` 改到 `Swizzle128` 之后收益明显，那 QK 这边是不是也可以如法炮制？我决定把 QK 的 swizzle 也改成 `Swizzle128`，这样 row stride = 256B 的情况下，128B 的 swizzle 跨度可以让 bank 分布更加均匀，理论上可以完全消除 QK 的 bank conflict。

### 代价呢

`Swizzle128` 对 shared memory 地址计算的要求更高一些，可能会多几条指令。但相比于消除 bank conflict 带来的收益，这个代价应该是可以接受的。而且 QK 的 `ldmatrix` 数量本身就不算多，多几条地址计算指令不会成为瓶颈。

### V3 的性能

改完 QK 的 swizzle 之后重新跑了一下：

```text
Elapsed Time (total):      0.0332 seconds
Average Time Per Step:     0.332 ms
Compute Performance:       103.58 TFLOPS
Memory Throughput:         188.42 GB/s
```

从 V2 的 91.46 TFLOPS 提升到了 103.58 TFLOPS，大概又提升了 13%。虽然还没到质变的程度，但还在提升。

从 profiling 数据来看，`memory_l1_wavefronts_shared` 从 V2 的大概 6000 万 sectors 降到了 5000 万左右，bank conflict 降低得到了证实。这个指标下降说明 shared memory 的有效利用率提高了，跟我预期的一致。

---
## V4: Softmax 优化 — 2-Row 处理

从 ncu report 来看，目前 warp state 里 short scoreboard stall 的占比比较高，大概 50% 左右，scheduler 占 27%，occupancy 只有 25%。算一下，每一个 SM 就只有 16 个 warp，平均到每个 scheduler 大概 3.6 个 warp，stall 的时候很难找到 ready 的 warp 来切换，这是个大问题。

先着手解决 short scoreboard stall。report 里的 detail 大概是这样的：

> On average, each warp of this workload spends 5.5 cycles being stalled waiting for a scoreboard dependency on a MIO (memory input/output) operation.

所谓 MIO 操作即：LDS, STS, SHFL, MUFU。那么谁在用这些指令呢？

| 阶段 | 用到的 MIO 指令 | 走不走 MIO scoreboard |
|------|----------------|----------------------|
| cp.async 加载 | — | 异步，走 long scoreboard / barrier |
| QK / PV MMA | LDSM（ldmatrix） | 走 |
| softmax | LDS（读 logits）+ SHFL×10（两轮归约）+ MUFU.EX2（__expf）+ STS（写 P）| 全占 |

看起来 softmax 是 MIO 指令的重灾区。为了确认 stall 主要发生在哪一段，我用了个脚本按源码行聚合采样：

```python
PYTHONPATH="......../ncu-report" python3 << 'EOF'
import ncu_report
from collections import defaultdict

r = ncu_report.load_report("ncu_flash_attn_learn_source.ncu-rep")
a = r.range_by_idx(0).action_by_idx(0)

METRIC = "smsp__pcsamp_warps_issue_stalled_short_scoreboard"
m = a.metric_by_name(METRIC)
cids = m.correlation_ids()

per_line = defaultdict(int)
total = 0
for i in range(m.num_instances()):
    addr = cids.as_uint64(i)
    samples = m.as_uint64(i)
    total += samples
    info = a.source_info(addr)
    if info:
        per_line[(info.file_name(), info.line())] += samples
    else:
        per_line[("<no-line-info>", addr)] += samples

for (f, line), s in sorted(per_line.items(), key=lambda kv: -kv[1])[:20]:
    print(f"{f}:{line}  samples={s}  ({100*s/total:.1f}%)")
print(f"\ntotal = {total}")
EOF
```

结果出来，softmax 里的 shuffle 占绝大多数。这其实不意外，shuffle 本身就要走 MIO scoreboard，而且 softmax 里每行要做两次 shuffle 归约，整个 block 有 Br=32 行，累积起来就是几千个采样点。

### 让 Warp 同时处理两行

既然 softmax 是瓶颈，那就针对它下手。一个合理的改动是：让每个 warp 同时处理两行。

原来的逻辑是一个 warp 处理一行，做 softmax 的时候等 shuffle 依赖、等 LDS、等 STS，warp 就卡在那里。改成两行之后，warp 可以在等待其中一行的依赖时，切到另一行继续干活，起到一种类似掩护的作用，把 stall 的时间利用起来。

另外，我顺便把所有 `[elementsPerLane]` 长度的局部数组改成了标量变量，这样或许能稍微缓解一下寄存器压力，进而改善 occupancy。

### V4 的性能

改动之后重新跑了一下：

```text
Elapsed Time (total):      0.0287 seconds
Average Time Per Step:     0.287 ms
Compute Performance:       119.76 TFLOPS
Memory Throughput:         217.84 GB/s
```

从 V3 的 103.58 TFLOPS 提升到了 119.76 TFLOPS，大概又涨了 15%。

从 ncu report 里的采样数据来看，变化比较明显：

| | 上一版 | softmax 2-row | 变化 |
|--|--|--|--|
| 总样本数 | ~11,360 | ~7,870 | -31% |
| shuffle 链相关 | 7,151 (62.9%) | 4,439 (56.4%) | -38% |
| LDS load-use | 1,474 (13.0%) | 850 (10.8%) | -42% |
| MMA (cuh:52 等) | 826 (7.3%) | 1,011 (12.9%) | 基本持平 |

shuffle 链相关采样从 62.9% 降到了 56.4%，绝对值降了 38%，说明 2-row 确实起到了作用。而且第二条链的延迟基本被第一条链的等待完全覆盖，符合预期——这印证了“掩护”的思路是有效的。

---
## V5: Padding — 解决 stmatrix 的 Bank Conflict

至此，我们已经稍微缓解了 warp short scoreboard stall 的问题。但在 ncu report 里还有一段比较扎眼的话：

```
OPT   Est. Speedup: 12.29%
This kernel has uncoalesced shared accesses resulting in a total of 6291456 excessive wavefronts (13% of the
total 49958912 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations.
The CUDA Best Practices Guide
(https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c
-ab) has an example on optimizing shared memory accesses.
```

好吧，ncu 直接告诉我这里有 12.29% 的潜在提升空间。

### 先算一下 Excessive Wavefronts 的来源

所谓 excessive wavefronts，简单来说就是 shared memory 操作没有充分利用带宽，产生了多余的 wavefront。我大概算了一下：

看 `stmatrix_f32` 这条指令，理想情况下是 32 lane × 8B = 256B，对应 2 个 wavefront。但如果没有引入任何 swizzle 或 padding，会发生 8-way bank conflict，导致实际需要 16 个 wavefront，excessive 占比算下来大概是 75%。


然后我把前文那个 python 脚本的 metric 改了一下，定位到 excessive wavefronts 具体发生在哪里——似乎结果全在 `stmatrix32` 里。换句话说，写回 shared memory 的那一步在大量产生 bank conflict。

### 为什么 stmatrix 会有问题

![stmatrix32](sts64_swizzle64_vs_swizzle128_layout.png)

`stmatrix32` 的写指令是 `STS64`，要求 8B 对齐。但 swizzle 的粒度是 16B，这个粒度对于 `stmatrix` 来说太大了，导致多个 lane 的写操作落到同一个 bank 上，产生冲突。

一个比较直接的思路是：不用 swizzle，改用 padding。通过在 shared memory 的 row stride 里插入一些 padding 元素，让相邻行的同一列错开 bank，从而避免冲突。

我试了两个 padding 值：

- padding 68：理论上可以消除一半的 conflict。
- padding 72：理论上可以完全错开，消除所有 conflict。

### V5 的性能

先说 padding 68：

```text
Elapsed Time (total):      0.0271 seconds
Average Time Per Step:     0.271 ms
Compute Performance:       126.66 TFLOPS
Memory Throughput:         230.39 GB/s
```

ncu 里 excessive sectors 降到了 524,288，比之前的 6,291,456 少了一个数量级。

再说 padding 72：

```text
Elapsed Time (total):      0.0264 seconds
Average Time Per Step:     0.264 ms
Compute Performance:       130.04 TFLOPS
Memory Throughput:         236.55 GB/s
```

这次 ncu 里 excessive sectors 直接降到了 0。

从 V4 的 119.76 TFLOPS 到 padding 72 的 130.04 TFLOPS，大概又提升了 8.6%。虽然提升幅度在逐渐变小，但这块本来也不是最大的瓶颈，能有这个收益已经不错了。

---

## G1收尾

前文系列1从 V1 到 V5，我们完成了以下几个关键优化：

| 版本 | 优化点 | TFLOPS | 提升 |
|------|--------|--------|------|
| V1 | Baseline (naive 实现) | 56.83 | — |
| V2 | PV Swizzle: None → 128B | 91.46 | +1.61x |
| V3 | QK Swizzle: 64B → 128B | 103.58 | +13% |
| V4 | Softmax 2-row 处理 | 119.76 | +15% |
| V5 | stmatrix padding 72 | 130.04 | +8.6% |

从 56.83 TFLOPS 到 130.04 TFLOPS，大概整体提升了 2.3 倍。虽然距离 baseline 的 1156 TFLOPS 还有很大差距，但每一步都有明确的方向和收益，这条路走得还算合理？

这五版把 shared memory 这一侧的问题基本解决了：excessive sectors 从 600 多万直接归零，bank conflict 不再是瓶颈。但 ncu 里还有两个挥之不去的数字：

- short scoreboard stall 依然占大头：V4 的 2-row 处理把 shuffle 链相关采样从 62.9% 压到了 56.4%，但 softmax 里 `LDS → SHFL → exp → SHFL → STS` 的串行链没被打破，warp 依然在大量等待 MIO 数据。
- occupancy 只有 25%：每个 SM 只有 16 个 warp，平均每个 scheduler 不到 4 个，stall 的时候找不到足够的 ready warp 来切换。

shared memory 优化完了，下一个问题在 softmax 的关键路径和 occupancy 上。G2 打算换个角度，从算术指令和 occupancy 入手，看看能不能再往前蠕动一点。
