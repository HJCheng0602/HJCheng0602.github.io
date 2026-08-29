---
title: "Flash Attention CUDA Kernel 优化: 从 56 到 986 TFLOPS on B300 — G2: 从算术指令到 Occupancy 的挣扎 (130→144)"
date: 2026-07-31 00:00:00
description: "A CUDA kernel optimization note for Flash Attention on B300. G2: arithmetic instructions & occupancy."
series: flash-attention-b300
series_order: 2
timeline_title: "G2 · 从算术指令到 Occupancy 的挣扎"
timeline_metric: "130 → 144"
tags:
    - CUDA
    - kernel optimization
    - flash attention
categories:
    - practice
---
![cover](cover.png)

## 背景


G1 的 V5 收尾，从 56.83 TFLOPS 爬到了 130.04 TFLOPS，翻了一倍多。几个比较明确的结论是：

1. bank conflict 基本解决：PV/QK 的 swizzle 改到 128B，加上 stmatrix padding 72，excessive sectors 从 600 多万降到了 0。
2. softmax 依然是瓶颈：2-row 处理虽然降低了 stall，但 short scoreboard 占比还是高，SHFL 串行链没被打破。
3. 微调收益递减：V5 之后再做小修小补已经看不到明显提升了，需要换个方向。

> 我看这bankconflict基本解决，softmax也基本清晰，主要链路基本完善，如果TFLOPS再高一些， 便是我的理想算子doge

所以 G2 打算换个方向，从算术指令和 occupancy 的角度再往前一点，看看能不能把 TFLOPS 从 130 往上再推一点。于是有了 V6。



## V6: 算术指令优化 — 一个 SOTA 启发的尝试

进入系列2，我打算尝试一些新的方向。这次参考了一些 SOTA 工作的思路，集中在算术指令优化上，想看看能不能从计算侧再搞出一些性能来。

### 改了什么呢？

V6 的改动主要集中在 softmax 的计算路径上，零零碎碎改了好几处。

第一处：推迟 scale 乘法

在之前的 softmax 实现里，从 shared memory 读出 logits 之后会立即乘以 scale：

```cpp
fp32_t score00 = logits[row0 * Bc + col0];
fp32_t score10 = logits[row1 * Bc + col0];
// ...
score00 *= scale;
score10 *= scale;
// ...
```

但 scale > 0 是一个常数，不一定要乘在 logits 上。我想到可以把 scale 推迟到指数运算时再做，exp2(score·scale·log2e - max·log2e) → exp2(fma(score, scale_log2, -max_log2))，scale 与 log2e 合并。这样省掉了对每个值单独乘 scale 的运算，还可以用上 FMA，理论上能省几条指令。

第二处：FMA 替换

ncu 里有一个 Optimization 提示一直挂着：

```
OPT   Est. Local Speedup: 47.47%
This kernel executes 1048576 fused and 19660800 non-fused FP32 instructions.
By converting pairs of non-fused instructions to their fused (higher-throughput) equivalent,
the achieved FP32 performance could be increased by up to 47% (relative to its current performance).
```

47.47% 的潜在提升，这个数字看着有点诱人。于是我把 softmax 里的核心计算改成了：

```cpp
max_scaled = new_max * softmax_scale_log2;
numerator = exp2f(fmaf(score, softmax_scale_log2, -max_scaled));
```

用一个 FMA 替代了之前的乘加分离，理论上能提升吞吐，这也是第一处修改的延伸。

第三处：rescale 因子复用

在 `wmmaPV.scale2()` 里，每个 warp 要自己算 `expf(old_max - new_max)`。但注意到每 16 行有两个列 warp，同一个 rescale 因子被 32 个 lane 各自重算了一遍，这显然是一种浪费。

我在新版里让每个 softmax 的 lane 0 把这个值算好，直接写到 `smem_row_rescale[Br]` 里，其他 lane 从 shared memory 读就行了。这样还可以把 `max_old` 数组给删掉（虽然引入了新的数组，但大小一样）。

第四处：跳过第一个 KV tile 的初始化

初始化的 `max` 和 `sum_exp` 数组，在第一个 KV tile 时其实没有历史状态需要合并，不需要读旧值。改成一上来就存第一份数据，省掉了一次初始化和一次 `__syncthreads()`。后面的 `exp` 操作也可以省去一部分。

代价是增加了一些指令数，但我当时觉得这点 tradeoff 是值得的。

### V6 的性能：大失败

跑了一下 benchmark：

```text
Elapsed Time (total):      0.0264 seconds
Average Time Per Step:     0.264 ms
Compute Performance:       130.05 TFLOPS
Memory Throughput:         236.57 GB/s
```

……跟 V5 的 130.04 TFLOPS 相比，基本没变。折腾了这么多，结果性能纹丝不动。甚至可以说这版就是原地踏步。

### 为什么没效果

先看 ncu。优化提示确实变了：

```
OPT   Est. Local Speedup: 40.71%
This kernel executes 3080192 fused and 13500416 non-fused FP32 instructions.
By converting pairs of non-fused instructions to their fused (higher-throughput) equivalent,
the achieved FP32 performance could be increased by up to 41% (relative to its current performance).
```

虽然理论上还有 40% 的提升空间，但注意 non-fused 指令从 19660800 降到了 13500416，说明我们确实减少了一些非 FMA 指令。但整体性能没变，说明这些算术指令本来就不是瓶颈。

scoreboard stall 那里仍然是 3.8 cycles per stall，基本没改善。

一个合理的解释是：当前 kernel 的 compute throughput 本来就没跑满，指令发射间隔很大，大多数时间都是 latency bound，而不是 issue bound。

具体到 softmax 的关键路径，大概是这样的：

```
LDS → [5步串行 SHFL max 链] → exp → [5步串行 SHFL sum 链] → STS
```

这条路径上的 SHFL 和 LDS/STS 是串行依赖的，算术指令的优化根本动不到这条链。我们改的那些 FMA、推迟 scale，都是在这条关键路径之外的“旁支”上做文章，对整体延迟没有影响。

### v6总结

V6 虽然性能没有提升，但让我想清楚了一件事：优化要打在关键路径上。如果瓶颈是一条 LDS → SHFL → ... → SHFL → STS 的串行链，那么优化这条链上的 SHFL 次数、减少 LDS/STS 的往返，才是有效的方向。在旁支上做 FMA 替换，应该是没有用的。

下一步我打算直接面对这条串行链，看看能不能在 softmax 的实现层面做更激进的改动。或许可以尝试 reduce 掉一些 SHFL 的次数，或者换一种数据流来打破串行依赖。

---
## V7: 4-Row Softmax — 对 Scoreboard Stall 有无作用？

优化不能绕着关键路径走。于是我把 V6 的所有改动回退了，重新回到 softmax 本身，这次做了个更激进的尝试：单 warp 处理 4 行。

### 设计思路

之前 V4 是 2-row，让 warp 可以在等待一行的时候去处理另一行。这次扩展到 4-row，理论上 4 行里只有一行会产生延迟，剩下 3 行可以“掩护”过去，进一步 hide latency。

但代价也很明显：寄存器压力会增大。原来 2-row 的时候每个 warp 需要维护两组 row 的状态，现在变成 4 组。如果按最直接的方式写，寄存器数量大概会从之前的水平翻到 36 个左右，这显然不太行。

尝试把 numerator 的存储复用了一下：

```cpp
numerator = __expf(score0[r] - new_max_value[r]);
tile_sum[r] = numerator;                // 用途一：进部分和
store_probability_scalar(...);          // 用途二：写 P
```

因为 numerator 在每一行计算中只需要保留一份，不需要把 4 行的 numerator 同时存下来。这样寄存器数量从 36 压到了 29 个左右，刚好卡在 occupancy 不掉档的边界上。

为什么这个边界这么重要？上一轮大概是 120 reg/thread × 128 thread/block = 15360 寄存器。如果一个 SM 上能驻留 4 个 block，那总寄存器用量是 61440。如果寄存器涨到 129/thread，一个 block 就需要 16512 寄存器，4 个 block 就是 66048，超过了硬件限制，SM 就只能调度 3 个 block。occupancy 从 4 降到 3，性能大概率会回退。

所以 29 个寄存器这个数字，是比较有设计的。

### V7 的性能

benchmark：

```text
Elapsed Time (total):      0.0253 seconds
Average Time Per Step:     0.253 ms
Compute Performance:       135.73 TFLOPS
Memory Throughput:         246.89 GB/s
```

从 V6 的 130.05 TFLOPS 提升到了 135.73 TFLOPS，大概又涨了 4.4%。

从 ncu 数据来看，softmax 相关的 stall 占比确实降了不少：

| | V6 | V7 | 变化 |
|--|--|--|--|
| shuffle 相关 stall | ~63% | ~47% | -16% |
| softmax 整体 stall | ~90% | ~72% | -18% |

shuffle 相关 stall 从 63% 降到了 47%，softmax 整体从 90% 降到了 72%。这说明 4-row 确实起到了“掩护”作用，让 warp 在等待数据的时候有更多其他行可以干活。

不过 72% 还是很高，softmax 依然是绝对的主力瓶颈。下一步大概要想想更根本的办法，比如能不能减少 SHFL 的串行链长度，或者干脆换一种 softmax 的实现方式？

---
## V8: Vec2 — 能解决 LDS 的问题吗？

V7 之后，我又仔细看了一遍 ncu 的逐行分析。除了 SHFL 之外，LDS 相关的代码行也在贡献 MIO stall，大概 2.5 cycles per instruction 的样子。而且 memory throughput 一直没跑满，说明访存路径上还有优化空间。

### 主要思路

之前的实现里，每个 warp 负责的 col 是 stride 32 的，也就是一个 warp 里的 32 个 lane 各自处理一列。这次我尝试把 stride 改成 2，让 warp 以 vec2 的方式去访问 shared memory。

这个改动本身其实不大，看起来就是让每个 lane 一次读两个元素，而不是一个。但我当时的想法是：虽然改动小，但地址计算能少一半，或许能带来一些边际收益。

另外，我把 V7 里的 numerator 优化给回退了——就是那个复用 numerator 存 tile_sum 和写 P 的小技巧。想看看 vec2 本身的收益有多大，会不会被回退带来的额外寄存器开销给吃掉。

### V8 的性能

跑了一下 benchmark：

```text
Elapsed Time (total):      0.0240 seconds
Average Time Per Step:     0.240 ms
Compute Performance:       143.37 TFLOPS
Memory Throughput:         260.79 GB/s
```

从 V7 的 135.73 TFLOPS 提升到了 143.37 TFLOPS，大概又涨了 5.6%。虽然单次提升不大，但加上前面的积累，从 V1 的 56.83 TFLOPS 到现在已经翻了 2.5 倍。

### 为什么有效

ncu 里有一个比较意外的发现：寄存器从 120 掉到了 96。

vec2 只需要一个地址计算而不是两个，编译器在寄存器分配上更宽松了。虽然我回退了 numerator 优化，但 vec2 带来的寄存器节省似乎比 numerator 带来的开销还要大，所以整体反而是赚的。

另外一个角度是 STS 操作。之前每写一个 fp16 元素，实际上是 2B × 32 lane = 64B，wavefront 的复用率很差。现在用 vec2 之后，一次 STS 是 4B × 32 lane = 128B，恰好能填满一个 wavefront。SASS 层面似乎也支持这个观察——虽然我没仔细反汇编，但从性能数据来看，这个解释是说得通的。

当然，这个优化本质上还是小修小补。V8 之后，softmax 在 ncu 里的占比依然很高，vec2 只是让 LDS 那块稍微好了一点，但 SHFL 链的问题还在。感觉接下来要做一些更结构性的改动了。

---

## V9: Stall 转移到了 stmatrix？

V8 之后重新跑了一下 py 脚本分析 stall 来源，发现最高项已经不在 softmax 里了——mma 在等 stmatrix。换句话说，瓶颈从 SHFL 链转移到了写回 shared memory 的路上。

这其实是好事，说明之前的 softmax 优化起了作用，瓶颈开始移动了。于是我把注意力转向了 `stmatrix` 这条路径。

### 分析 MMA 的执行路径

整个 mma 的执行路径大概是这样：

```
地址 IMAD ──→ LDSM×3 ──(~30拍 MIO 延迟)──→ HMMA×2 ──→ 下一个 kt
                    ↑可压缩              ↑（趟数固定）       ↑(38%)
```

看起来唯一能压缩的是地址运算那块。于是我想了三种方案：

| 方案 | 思路 | 代价 |
|------|------|------|
| qkfragpipe | kt 算着，kt+1 的货已在路上，延迟被 HMMA 遮住 | +一套 frag 寄存器 |
| qkaprefetch | 只预取 A（半个双缓冲） | +半套寄存器 |
| qkspecialized | 不遮延迟，只把 LDSM 发得更早（砍地址链） | 零寄存器 |

只做 QK 的原因比较直接：stall 里给的都是 QK 的 mma 代码段，PV 那边前面有一堆 softmax 可以遮蔽，暂时不需要动。

### 方案一：qkfragpipe（双缓冲）

顾名思义，对 QK 的 fragment 做双缓冲，让下一轮的数据提前加载好，延迟被 HMMA 掩盖。

跑了一下 benchmark：

```text
Elapsed Time (total):      0.0250 seconds
Average Time Per Step:     0.250 ms
Compute Performance:       137.57 TFLOPS
Memory Throughput:         250.24 GB/s
```

从 ncu 来看，mma stall 从 38% 降到了 9.4%，效果确实有。但 active warps 掉到了 3.69，因为双缓冲多占了一套寄存器，occupancy 下来了。这个 tradeoff 太大了，bench 性能反而比 V8 倒退了。

### 方案二：qkaprefetch（只预取 A）

这次只预取 A 矩阵，寄存器开销只有方案一的一半。

benchmark：

```text
Elapsed Time (total):      0.0249 seconds
Average Time Per Step:     0.249 ms
Compute Performance:       137.75 TFLOPS
Memory Throughput:         250.57 GB/s
```

比双缓冲略好一点，但 active warps 仍然不到 5。说明 occupancy 的影响还是占绝对主导地位。

### 方案三：qkspecialized（砍地址链）

这次不动寄存器，只优化地址计算。

5.6 大人给了一个定论：generic forward 里每个 kt 步、每次加载都要算一遍完整的坐标变换链，开销确实不小。原来的逻辑大概是：

```cpp
local_m = (lane & 0xF) + (m << 4);           // 逻辑行
local_n = ((lane >> 4) & 0x1) + (kt << 1);   // 逻辑向量列
coord = coordinate_swizzle_2d<swizzle>(local_m, local_n); // 通用2D swizzle
addr = A + coord.x * WarpK + coord.y * VPT;  // 展平
```

每个 kt 步 3 次加载 × 8 个 kt = 每 warp 每 tile 24 条完整的坐标→swizzle→展平链。优化之后变成了：

```cpp
k_vector = kt << 1;                    // 移位（常量折叠）
a_pv = (a_sel + k_vector) ^ xor_bits;  // 1加1异或
b_pv = (b_sel + k_vector) ^ xor_bits;  // 1加1异或
```

从大约 20 条整数指令/kt 压到了 6 条左右，而 active warps 还是原来的那样。

benchmark 结果：

```text
Elapsed Time (total):      0.0239 seconds
Average Time Per Step:     0.239 ms
Compute Performance:       143.52 TFLOPS
Memory Throughput:         261.06 GB/s
```

确实回到了 V8 的水平（甚至略高一点，143.52 vs 143.37），但提升并不显著。ncu 里 warps 回到了 5，说明 occupancy 保住了，但对于整个关键路径的改善有限。

### 小结

三种方案跑下来，结论大概是：

1. 双缓冲能解决 mma stall，但 occupancy 掉得太狠，得不偿失。
2. 预取 A介于中间，但同样受限于寄存器压力。
3. 专用地址计算最干净，保住了 occupancy，但收益不明显，关键路径的延迟没有本质改变。

于是暂时决定在这个方向先放一放。可能后面等 occupancy 有了更多余量之后，再回来继续深挖 qkfragpipe 这条路。目前的收益已经不足以 justify 继续在这个方向上投入了。

---

## V10: 回到算术指令优化 — 我再也不相信 FMA 相关的提示了

V9 的尝试虽然没有大幅提升，但 ncu 里有一条优化提示一直存在：

```
OPT   Est. Local Speedup: 47.47%
This kernel executes 1048576 fused and 19660800 non-fused FP32 instructions.
By converting pairs of non-fused instructions to their fused (higher-throughput) equivalent,
the achieved FP32 performance could be increased by up to 47% (relative to its current performance).
```

47.47%，都快一半了。虽然 V6 已经在这方面失败了一次，但我总觉得那次没做到位——也许是我改的方法不对，也许这次换一种方式就能拿到这个收益？


### 重新审视 Softmax 的计算流

V6 的时候我确实做了一些 FMA 替换，但当时是在整体架构没动的情况下硬换的。这次我重新梳理了一个 lane 拿到一对 (s0, s1) 之后的完整计算序列：

```text
LDS 64                  // 读 logits
FUL s0 = s0 * scale     // 乘 scale
FUL s1 = s1 * scale     // 乘 scale
FMNMX x (1 + 5) + SHFL x 5  // 找 max
FADD t0 = s0 - m        // 减 max
FADD t1 = s1 - m        // 减 max
FMUL u0 = t0 * log2e    // 乘 log2e
FMUL u1 = t1 * log2e    // 乘 log2e
MUFU.EX2 x 2            // exp2
```

这里面乘法和加法是分开的，确实有不少机会可以用 FMA 来合并。于是我做了一组比较系统的改动：

```cpp
const fp32_t softmax_scale_log2 = rsqrtf((fp32_t)HeadSize) * Log2E;

score0[r] = score_pair[r].x;
row_max[r] = new_max_value[r] * softmax_scale_log2;
numerator0[r] = exp2f(__fmaf_rn(score0[r], softmax_scale_log2, -row_max[r]));

history_scale = exp2f((old_max - new_max) * softmax_scale_log2);
new_sum = __fmaf_rn(old_sum, history_scale, tile_sum[r]);

// scale_pv_history 内部同样用 exp2f((max_old - max) * α₂)
```

核心思路是把 `score * scale` 和 `t - m` 这些操作都合并到 FMA 里，`exp2f` 的输入也直接用 FMA 算出来。原本需要多条独立指令的地方，现在一条 FMA 就能搞定。

### V10 的性能


```text
Elapsed Time (total):      0.0240 seconds
Average Time Per Step:     0.240 ms
Compute Performance:       143.37 TFLOPS
Memory Throughput:         260.79 GB/s
```

……跟 V8/V9 基本一致。折腾了一圈，性能纹丝不动。

从 ncu 来看，那两条优化提示确实变了——non-fused 指令少了，fused 指令多了。但整体性能没变。

### 总结

ncu 的 Local Speedup 估算是基于“当前 compute 是瓶颈”这个假设的。如果 kernel 实际上是 memory-bound 或者 latency-bound，那就算把 FP32 指令全部换成 FMA，也见不到任何收益。

当前 softmax 的关键路径是：

```text
LDS → SHFL(max) → exp → SHFL(sum) → STS
```

这条链上的主要延迟来自 SHFL 的往返和 LDS/STS 的访存，FMA 只能优化中间的算术部分，对这条链的端到端延迟几乎没有影响。

我再也不相信 FMA 相关的提示了。www


---

## V11: 消融实验 — 为什么加回来的优化反而变差了？

V10 的 FMA 尝试虽然失败了，但我想着 V6 里还有两个优化点当时是跟 FMA 一起做的，后来回退的时候一并回退了。也许它们本身是有效的，只是被 V6 的其他改动掩盖了？

于是我决定把 V6 里的两个改动单独加回来试试：

1. 推迟 scale 乘法：把 `score * scale` 推迟到指数运算时做。
2. rescale 因子复用：lane 0 算好 rescale 因子写到 shared memory，其他 lane 复用。

理论上这两个改动都是“省计算”的，应该不会有副作用才对。

### V11 的性能

跑了一下 benchmark：

```text
Elapsed Time (total):      0.0251 seconds
Average Time Per Step:     0.251 ms
Compute Performance:       137.00 TFLOPS
Memory Throughput:         249.21 GB/s
```

？从 V10 的 143.37 TFLOPS 掉到了 137.00 TFLOPS，反而降了 4.4%。

有意思，按理说省了几条指令不应该变慢才对。我查了一眼 ncu 的 register 用量，发现问题了：`_no_init` 引入了两个 softmax 模板，寄存器占用从 96 涨到了 122。虽然 single-thread 的指令数少了，但 occupancy 直接掉了一档，active warp 数减少，整体吞吐反而下降了。

### 做个消融

为了确认是这个原因，我单独把寄存器相关的那部分改动去掉，只保留算术逻辑的优化，重新跑了一遍：

```text
Elapsed Time (total):      0.0238 seconds
Average Time Per Step:     0.238 ms
Compute Performance:       144.41 TFLOPS
Memory Throughput:         262.68 GB/s
```

果然，回到 144.41 TFLOPS 了。消融实验证实了问题不在算术优化本身，而是那两个 softmax 模板造成了spill。

### 教训

V11 让我意识到一件事：单个优化点要放在整体里看。把 V6 的两个改动单独拿出来，它们本身是有效的——省了计算，没有其他副作用。但当它们和 V10 的状态组合在一起时，模板展开导致了寄存器爆炸，反而得不偿失。

有些优化需要“一起上”才有效，有些则需要“分开上”才能看清效果。V11 至少证明了那两条改动本身是有价值的，只是需要小心控制模板展开带来的寄存器开销。

下一步大概要在保持寄存器不spill的前提下，再考虑怎么把这些算术优化塞回去。

---

## G2 小结

到 V11 为止，G2 这一系列的优化算是告一段落了。回看整个迭代过程，从 V6 的失败尝试，到 V7/V8 的小步前进，再到 V9 的地址优化、V10 的 FMA 执念、V11 的消融实验，每一步都留下了比较明确的结论。

### 性能演进回顾

把 G2 系列各个版本串起来看：

| 版本 | 核心改动 | TFLOPS | 变化 |
|------|---------|--------|------|
| V5（G1 终点） | stmatrix padding 72 | 130.04 | — |
| V6 | FMA + rescale + 跳过初始化 | 130.05 | +0.01 |
| V7 | 4-row softmax | 135.73 | +4.4% |
| V8 | vec2 访存 | 143.37 | +5.6% |
| V9 | qkspecialized 地址优化 | 143.52 | +0.1% |
| V10 | FMA 再次尝试 | 143.37 | -0.1% |
| V11 | 消融 + rescale 复用 | 144.41 | +0.7% |

从 V5 的 130.04 TFLOPS 到 V11 的 144.41 TFLOPS，G2 整体大概提升了 11%。虽然跟 G1 那种动辄翻倍的进步没法比，但在这个阶段每提升一点点都可以学到很多？

### 一些不那么显然的认知

这个系列跑下来，除了性能数字本身，我大概还学到了几件事：

1. 优化提示的“潜力”不等于“收益”：ncu 说 FMA 能带来 47% 的提升，但那是假设 compute 是瓶颈的前提下。如果 kernel 是 latency-bound，算术优化就是在隔靴搔痒。现在我学会了选择性忽略某些提示。

2. Profiling 要看到“谁在等谁”：只看整体时间和 TFLOPS 是不够的。从 SHFL → LDS → stmatrix → mma，瓶颈在移动，每次移动都对应着上一轮优化的成功。这个“移动”的过程本身比最终数字更有信息量。

3. 寄存器是比 shared memory 更稀缺的资源：V9 的双缓冲方案在技术上是有效的——mma stall 从 38% 降到了 9.4%。但因为多占了一套寄存器导致 occupancy 掉了，最终性能反而更差。在 B300 上，寄存器压力往往比 shmem 容量更早成为天花板。

4. 小优化叠加的收益会递减：V7 的 4-row、V8 的 vec2 都还有可见收益，但到了 V9/V10/V11，收益已经基本在噪声水平了。这意味着 G2 已经接近了当前架构下的局部最优，再做微调的意义不大了。

### 下一步

G2 到这里就差不多了。144 TFLOPS 虽然离 baseline 的 1156 TFLOPS 还很远，但至少每一步都留下了可以追溯的记录。

回看 G2 整个过程，softmax 这条链已经修到了极限：SHFL 从 5 步串行压到 2-row、4-row 掩护，LDS 用 vec2 减半，算术指令确认不是瓶颈。V9 里瓶颈转移到了 stmatrix——P 算完要写回 shared memory，PV 才能从那里读到。这让我意识到一个问题：当前的 wmma 布局下，一行 Q 的 logits 被两个 warp 各管一半（TileN=2），P 不走 smem 就交不到 PV 手里。

换句话说，这段时间在 softmax 上的所有努力，都是在优化"数据怎么搬"，但"数据必须搬"这件事本身，可能才是真正的墙。接下来该动的是 wmma 的布局本身——让一个 warp 管一整行，P 留在寄存器里直接做 softmax，试试能不能把这堵墙拆掉。

G2 到此结束。

---
