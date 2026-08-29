---
title: "Flash Attention CUDA Kernel 优化: 从 56 到 986 TFLOPS on B300 — G4: SM103 + TMA + TCGen05 (227→221)"
date: 2026-07-31 00:00:00
description: "A CUDA kernel optimization note for Flash Attention on B300. G4: SM103, TMA, TCGen05."
series: flash-attention-b300
series_order: 4
timeline_title: "G4 · SM103 + TMA + TCGen05"
timeline_metric: "227 → 221"
tags:
    - CUDA
    - kernel optimization
    - flash attention
categories:
    - practice
---
![cover](cover.png)

## 背景


G3 从 V12 到 V13，核心是架构级改动——fragment reshape 和 warp specialization。V12 的 Br 32→64 直接带来了 52% 的提升，V13 的 warp specialization 又往前蠕动了 3.5%，最终停在 226.83 TFLOPS。

G3 结束时大概有这么几个结论：

1. 架构级改动的收益远大于微调：V12 一个改动顶 G2 六个版本。
2. Br=64 在当前架构下是个舒服的点：Br=128 会 spill，Br=32 复用不够。
3. cp.async 的收益边际在缩小：下一步可能要上 TMA。
4. static smem 48KB 限制已被突破：V13 用了 58KB，在 SM103 上能跑。

但有一个问题之前一直被忽略了：我们用的是 B300，是 SM103，不是 SM80。之前所有优化都是基于 SM80 的 mma 指令和 ldmatrix 写的，完全没用到 SM103 的新特性。

于是 G4 决定换个赛道——用上 SM103 ：TMA + TCGen05。


## V14: 从 SM80 到 SM103 — TMA + TCGen05 初版

之前多次优化最后只能达到 torch sdpa 的 20%，显然我们注意到——我们用的是 SM103 而非 SM80。于是决定用上 `tcgen05`，来写一版新算子。

这里有一个让我很开心的地方：之前读 cutlass 一直读不懂，但现在看 ptx 文档感觉清晰很多。我觉着或许学习 tensorcore 从 ptx 开始是有道理的？难道我真的学会了写算子？不管了，先写出来再说。

这一版其实与前面的 SM80 实现大不相同，基本上可以视为一次重新实现。我们直接从大体思路讲，细节层面的 ptx 指令就不逐条展开了。

### 改动点

为了保证每步跨度不那么太大，我先只改了一小部分，主要集中在 QK 侧：

a. TMA 替换 cp.async

原先的 Q/K/V 加载用的是 `cp.async`，现在改成了 TMA（Tensor Memory Accelerator）。好处是：虽然也是异步加载，不走寄存器，但 swizzle、mbarrier、地址计算这些东西都可以放到 DMA 硬件代劳，不需要在软件里手算地址偏移了。

代价是 TMA 的 descriptor 配置比较复杂，一旦配置错了，debug 起来相当痛苦。

b. QK 从 mma 改成 tcgen05

原先的 SM80 `mma.sync` 指令是同步的，4 个 warp 发一堆指令才能算完一个 64×64 的 tile。现在 `tcgen05` 可以走异步，而且一条指令就能算 128×128，不再需要 `ldmatrix`，不需要经过寄存器。

这意味着 QK 阶段的指令数量会大幅减少，理论上效率会高很多。

c. 结果存在 TMEM 里

原先 P 存在寄存器里（或者 smem 里），现在 `tcgen05` 的结果存在 TMEM（Tensor Memory） 里。这个加载的延迟和带宽特性跟寄存器不太一样，感觉不太好直接比较优劣，先跑起来再说。

d. Softmax 的改变

现在 P 存在 TMEM 里，逻辑上每一行对应一个 lane。于是 softmax 改成了单线程处理一行的模式——每个 thread 负责一行的全部 64 列。这样保存 P 的时候可以用 128-bit 宽指令写出去，但性能肯定比不上寄存器里做 softmax 那么快，算是一个 tradeoff。

e. PV 没怎么动

PV 这边暂时保持原样，先把 QK 侧跑通再说。


### 两个坑

验证手写 ptx 正确性的过程中，踩了两个比较典型的坑，记录一下。

坑一：tcgen05.ld 的地址理解错误

我在调用 `tcgen05.ld.sync.aligned.32x32b.x8.b32` 的时候，一开始是这么写的：

```cpp
asm volatile(
    "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
    : "=r"(values[0]), ...
    : "r"(tmem_address)
    : "memory"
);
```

每个 warp 调用的时候，我传的是：

```cpp
ptx::tcgen05_load_32dp32b8x(tmem_base + vector * 8, values);
```

但实际上 tmem base 的选址是由头 32 和尾 32 共同决定的。也就是说，你需要对不同 warp 提供不同的行地址，而不是简单地 `vector * 8` 就能搞定。

坑二：TMA 与 TCGen05 的 SMEM 布局不匹配

原来的 shared-memory staging 是：

- SMEM 逻辑布局：`[M][K=128]`，row stride = 256B
- TMA 写到 shared memory 后，Q/K 的 128B block 排列顺序是 `[M][K]` 的线性布局

但 `tcgen05.mma` 在 K-major + SW128 模式下，通过 SmemDescriptor 解释这些 block 的排列顺序，和 TMA 的写入顺序不一致。

具体来说，我的 TensorMap 是这样的：

```cpp
uint64_t global_dimensions[4] = { 64, 2, NumHeads, SeqLen };
uint32_t box_dimensions[4] = { 64, 2, 1, Br };
```

一个 logical row 是 `64 × 2 half = 128 half = 256B`。TMA 搬完到 SMEM 以后，从矩阵角度来看，SMEM 里是这样的：

```
row 0: K[0:64] K[64:128]   128B    128B
row 1: K[0:64] K[64:128]   128B    128B
row 2: ...
```

但 `tcgen05` 的 SmemDescriptor 是这样配置的：

```cpp
cute::UMMA::SmemDescriptor make_k_major_descriptor(void* pointer) {
    descriptor.layout_type_ = LayoutType::SWIZZLE_128B;
    descriptor.leading_byte_offset_ = 0;
    descriptor.stride_byte_offset_ = (8 * HeadSize * sizeof(half)) / 16;
}
```

对于 K-major + FP16 + SW128，8-row group 内部的 M 方向布局是由 `tcgen05` 的 canonical layout 固定的，它会按照类似这样的方式来读：

```
row0 base = SMEM + 0B
row1 base = SMEM + 128B
row2 base = SMEM + 256B
row3 base = SMEM + 384B
...
```

而 TMA 写的布局是每行 256B，两个 K 段各 128B。

这个 bug 在 `advance_descriptor_k()` 里也会触发：

```cpp
for (int32_t k_block = 0; k_block < HeadSize / UmmaK; ++k_block) {
    uint64_t q_descriptor = advance_descriptor_k(descriptor_sQ, k_block);
}
```

地址推进为：

```
k_block 0: +0B
k_block 1: +32B
k_block 2: +64B
k_block 3: +96B
k_block 4: +128B  ← 从这开始就错了
k_block 5: +160B
...
```

修复方法是在 TMA 加载的时候就把数据拆成两个部分来加载，让 SMEM 里的实际布局和 `tcgen05` 的 canonical layout 对齐。也就是说，TMA 写的时候就要按照 `tcgen05` 预期的排列方式来写，而不是按照自然的 `[M][K]` 布局。


### V14 的性能

这一版的主要目标是验证手写 ptx 的正确性，性能倒在其次。双缓冲、流水线什么的都还没上，所以预期会比较烂。

跑了一下：

```text
Elapsed Time (total):      0.0191 seconds
Average Time Per Step:     0.191 ms
Compute Performance:       179.81 TFLOPS
Memory Throughput:         327.08 GB/s
```

嗯，从 V13 的 226.83 TFLOPS 回退到了 179.81 TFLOPS。但说实话，性能回退没有想象中那么严重——毕竟这是一个完全重写的版本，很多优化都还没做，能到 180 TFLOPS 已经比我想象中好了。

ncu 里 long scoreboard 和 barrier 占比很高，符合预期——因为现在用的是异步的 TMA 和 tcgen05，需要大量依赖 mbarrier 做同步，这部分开销还没优化。

V14 算是把 SM103 的管线跑通了，下一步就是在这个基础上做流水线和双缓冲。

---

## V15: PV 迁移到 B300 — 完整 TCGen05 管线

V14 只把 QK 搬到了 tcgen05 上做，PV 还是用的 SM80 的老路子。这版把 PV 也迁过来了，整条管线都用 SM103 的指令跑。

### 流程

大致流程是这样的：

1. QK 用 tcgen05.mma 算，结果直接留在 TMEM 里。
2. 从 TMEM 读 P，做 online softmax，结果写到 smem_P。
3. 复用 TMEM 空间，用 tcgen05 做 PV matmul。
4. 最终的 accumulator 放到寄存器 fragment 里。

整体还是上一版的改进版，没有做架构层面的大调整，主要就是把 PV 那一段的指令替换成 tcgen05 版本。

### V15 的性能

跑了一下：

```text
Elapsed Time (total):      0.0188 seconds
Average Time Per Step:     0.188 ms
Compute Performance:       182.50 TFLOPS
Memory Throughput:         331.97 GB/s
```

从 V14 的 179.81 TFLOPS 提升到了 182.50 TFLOPS，大概涨了 1.5%。有提升，但不多。

从 ncu 来看，PV 那边的指令确实换成了 tcgen05，但整条管线还是单 stage 的，没有做流水线，也没有双缓冲。再加上 accumulator 放到寄存器之后出现了 spill，性能低是非常正常的。

这版的核心目标还是验证 PV 用 tcgen05 的正确性——能跑通、数值正确、不崩，就算成功。性能优化的事情，等管线稳定了再说。

下一步应该要开始做 pipeline 了，把 TMA 加载、tcgen05 计算、softmax、PV 这些阶段重叠起来，才能把 SM103 的吞吐真正发挥出来。

---


## V16: 寄存器优化 — 终于把 spill 摁住了

V15 虽然把 PV 也迁到了 tcgen05，但 ncu 里出现了一个不太妙的现象：寄存器 spill。之前的版本寄存器一直压得比较好，现在反而倒退了，说明 PV 的读回阶段确实有点问题。

### 问题定位：PV 读回的瞬时寄存器占用

具体来说，上一版 PV 读回是这样做的：

```cpp
uint32_t pv_values[32];
tcgen05_load_32dp32b32x(..., pv_values);  // 一条指令，32 个目的寄存器同时活跃
tcgen05_wait_load();                      // 32 个值一起 rescale + 累加
```

`tcgen05_load_32dp32b32x` 一条指令读 32 个值，占 32 个寄存器。加上其他正在存活的变量，寄存器峰值一下子就上去了，直接触发 spill。

这一版改成分批读：

```cpp
for (vector = 0; vector < 4; ++vector) {
    uint32_t pv_values[8];                // 瞬时活跃值 32 → 8
    tcgen05_load_32dp32b8x(tmem_base + pv_column_base + vector * 8, pv_values);
    tcgen05_wait_load();                  // 8 个值就地 rescale 进 output_accumulator
                                          // pv_values 立即死亡
}
```

每次只读 8 个值，处理完就释放，寄存器峰值从 64 降到了 40。这个改动的效果是立竿见影的。

### 另一个小优化：smem descriptor 按需构造

上一版还有一个问题：`tcgen05` 的 smem descriptor 是在所有 thread 里构造的，但实际上只有一个 warp 消费这些数据。其他 warp 白占了一堆寄存器。

这版把 descriptor 的初始化移动到真正 issue tcgen05 的那个 warp 内，省去了其他 warp 的寄存器开销。代价是每一个 seqtile 都要重做一遍，但这点开销相比寄存器省下来的收益，完全可以忽略。

### V16 的性能

跑了一下 benchmark：

```text
Elapsed Time (total):      0.0171 seconds
Average Time Per Step:     0.171 ms
Compute Performance:       201.13 TFLOPS
Memory Throughput:         365.85 GB/s
```

从 V15 的 182.50 TFLOPS 提升到了 201.13 TFLOPS，大概涨了 10%。ncu 里寄存器降到了 96，spill 消失了，符合预期。

不过 ncu 里还有一个数据让我有点在意：TC 利用率只有 6.7%。显然 pipeline 串行的问题仍然没有改善，计算单元大部分时间在等数据。

### 消融：单独测一下分批读回的粒度

因为上一版是同时做了两个改动（分批读回 + descriptor 按需构造），我有点好奇单独看分批读回这个优化本身有多大的收益。于是试了一下中间版本，只改分批读回，不改 descriptor：

l32（每次读 32 个值，分批处理）：

```text
Elapsed Time (total):      0.0180 seconds
Average Time Per Step:     0.180 ms
Compute Performance:       190.56 TFLOPS
Memory Throughput:         346.64 GB/s
```

比 V15 好一点，但比 V16 差。说明 descriptor 按需构造那部分也是有贡献的。

### 意外发现

在测试更小粒度的时候，我本来想试一下每次只读 16 个值（i16）。结果遇到了一件事：编译器把两个 load16 合并成了一条发射。这可不行，本来就是为了降寄存器峰值才拆开的，编译器这一合并等于白拆了。

最后强制加了 `asm volatile("bar.warp");` 才解决了这个问题。跑出来的结果：

```text
Elapsed Time (total):      0.0168 seconds
Average Time Per Step:     0.168 ms
Compute Performance:       204.04 TFLOPS
Memory Throughput:         371.15 GB/s
```

嗯，比 V16 又高了 1.5%，但提升已经不大了。

### 小结

V16 把寄存器 spill 的问题解决了，性能回到了 200 TFLOPS 以上。但 6.7% 的 TC 利用率说明一个问题：目前的瓶颈不在计算，也不在寄存器，而在 pipeline 太串行了。

TMA 加载、tcgen05 计算、softmax、PV 这些阶段基本是顺序执行的，没有重叠。下一步得搞 pipeline 了——把 TMA 的异步加载和 tcgen05 的计算叠起来，让 DMA 和 tensor core 同时干活，TC 利用率才有可能往上走。

---

## V17: Two Pass PV — 一条走不通的路

V16 虽然把寄存器 spill 摁住了，但 ncu 里 TC 利用率只有 6.7%，这个数字一直让我很在意。tensor core 大部分时间在摸鱼，说明计算远远不是瓶颈。

于是我开始重新审视 PV 阶段的访存模式。

### 问题分析

当前的 PV 流程大概是这样的：

1. QK 算完，P 在 TMEM 里。
2. 从 TMEM 读 P，做 softmax，写到 smem_P。
3. PV 阶段：从 smem_P 读 P，从 TMEM 读 V，做 mma，累加到 accumulator 寄存器里。
4. 每个 KV tile 循环都要做一次 rescale（因为 max 在变）。
5. 最终 accumulator 里存的是累加结果，占 32 个寄存器。

这里有一个隐含的开销：每个 KV tile 循环都要从 TMEM 里读一次 PV 结果，然后做 rescale。rescaling 本身不算贵，但每次读回 TMEM 再写回去的往返，加上 rescale 的同步等待，累积起来就是个不小的开销。

### Two Pass 思路

既然 rescaling 的根源是 max 在逐 tile 变化，那能不能让 max 不变？

思路是这样的：

1. Pass 1：完整跑一遍 QK + softmax，求出每一行的最终全局 max（也就是所有 KV tile 里最大的那个 max）。
2. Pass 2：用这个固定不变的全局 max，重新算一遍 QK，得到 P，然后直接做 PV。

关键在于：Pass 2 里的 max 是固定的，不需要 rescale，也不存在历史累积需要修正。每个 tile 的 P 可以直接累加到最终的 accumulator 上，而这个 accumulator 不需要存在寄存器里——可以直接放在 TMEM 里，让硬件来做累加。

这样一来：

- rescale 整个消失
- 跨 tile 更新退化为纯加法
- TMEM 硬件累加器接手
- acc 不再占寄存器
- 全程留 TMEM，最后只读回一次

听起来很完美

### V17 的性能

```text
Elapsed Time (total):      0.0195 seconds
Average Time Per Step:     0.195 ms
Compute Performance:       175.88 TFLOPS
Memory Throughput:         319.93 GB/s
```

……从 V16 的 204 TFLOPS 掉到了 175.88 TFLOPS，不但没涨，反而倒退了 14%。

### 为什么没效果

看 ncu 数据，tensor core 利用率从 6% 涨到了 8%——几乎没变化。说明问题根本不在重用次数上，而在其他地方。

再看 stall 分布，long scoreboard stall 都集中在 mbarrier 上。也就是说，TMA 加载的同步开销才是真正的瓶颈。two pass 方案相当于把 QK 多算了一遍，增加了计算量，但对隐藏 TMA 延迟没有任何帮助——因为 pipeline 还是串行的，TMA 的异步性完全没有发挥出来。

更麻烦的是，这个方案的 smem 占用到了 130 KB，单 CTA 就占满了，导致一个 SM 只能驻留一个 CTA。occupancy 直接跌到谷底。

### 教训

V17 的失败说明：数据复用（reuse）不是解决问题的万能药。如果瓶颈是访存延迟（TMA 加载），那么增加计算量反而会放大问题。只有在计算已经是瓶颈的时候，reuse 才有意义。

当前的真正问题是 pipeline 太串行——TMA 加载、tcgen05 计算、softmax、PV 这些阶段基本在排队，没有重叠。只要 pipeline 还是串行的，TC 利用率就上不去，再怎么折腾 rescale 和寄存器都没用。

所以接下来的方向不是做数据复用，而是做流水线重叠——让 TMA 加载下一组数据的时候，上一组的计算还没结束，把 TMA 延迟藏住，而不是等它完成再动。

---


## V18: Leader Wait — 一个看似合理实则翻车的优化

V17 走不通，回退到 V16 的状态。但在回退之后重新看 ncu 数据时，我注意到一个现象：stall 基本上都卡在 mbarrier 上。

### 直觉

mbarrier 是 TMA 异步加载的同步机制，每个 warp 在继续干活之前都要等 mbarrier 到位。当前实现是 warp 内所有 thread 都调用 `mbarrier.wait()`，但问题是——真的需要 32 个 thread 都去抢这一个 barrier 吗？

直觉上，每个 warp 只需要一个 thread 去 poll barrier 就行了，其他 thread 可以继续做别的事（或者至少不参与 LSU/MIO 的竞争）。我猜想当前的实现里，32 个 thread 同时涌向 mbarrier，挤占了 LSU 端口，反而增加了开销。

于是改了一版：每个 warp 只让 leader thread（lane 0）去调 `mbarrier.wait()`，其他 thread 等待 leader 广播结果。

### V18 的性能

```text
Elapsed Time (total):      0.0225 seconds
Average Time Per Step:     0.225 ms
Compute Performance:       152.96 TFLOPS
Memory Throughput:         278.24 GB/s
```

从 V16 的 204.04 TFLOPS 掉到了 152.96 TFLOPS，倒退了 25%。

这个数字让我有点不敢相信，反复排查了好几遍，确认是真实数据。

### 为什么翻车了

回看 ncu，确实 mbarrier 相关的 stall 取样下降了——说明 leader wait 本身是有用的。但代价是什么呢？

引入了一个问题：warp 内分化 + 额外的 sync 开销。

之前所有 thread 一起 poll barrier 的时候，虽然竞争存在，但 warp 是 uniform 的，没有 divergence。改成 leader-only 之后，其他 thread 在等广播的时候，warp 内部出现了分化，又需要额外的 `__syncwarp()` 来保证数据同步。这一步的开销比省下来的 LSU 竞争还要大。

简单来说：减少了 MIO 的竞争，但引入了 warp 同步的代价，后者超过了前者。

### 教训

这个优化方向本身没错——减少 mbarrier 的争抢是有意义的。但实现的粒度不对。leader wait 的思路在 warp 层面引入了新的 overhead，反而得不偿失。

如果要继续优化 mbarrier 的开销，可能需要在更粗的粒度上做（比如让一个 warp 专门负责同步），或者用 `mbarrier.try_wait` + 主动 yield 来替代阻塞式的 `wait`。但这些都牵涉到更大的架构调整，不是简单改一两行能搞定的。

暂时先把这个方向放一放，继续看其他瓶颈。

---

## V19: tcgen05.ld.red — 一个看着很好实际翻车的优化

V18 翻车之后，我回退到 V16，重新审视还有哪些地方可以动。

这次我把目光投向了 softmax 的求 max 阶段。当前实现里，每个 thread 从 TMEM 里单线程串行 load 出一行 P，然后逐个比较找 max。这个模式有以下几个问题：
1. 每条 load 都要走一遍 TMEM 访存
2. 比较是串行的
3. 每个 thread 各读各的，没有利用 TMEM 的广播/规约能力

翻 ptx 文档的时候，我发现了一个之前没注意到的指令：`tcgen05.ld.red`。它可以在加载的同时直接做 reduction——支持 max、min、sum 等操作。

这不就是为 softmax 量身定制的吗？原先需要“load → 比较 → load → 比较”的串行链，现在一条指令就能搞定。理论上能省掉不少指令和寄存器。

于是我只做了这一处修改，其他地方完全没动，想看看纯替换指令的效果。


### V19 的性能

```text
Elapsed Time (total):      0.0192 seconds
Average Time Per Step:     0.192 ms
Compute Performance:       178.71 TFLOPS
Memory Throughput:         325.06 GB/s
```

从 V16 的 204 TFLOPS 掉到了 178.71 TFLOPS，倒退了 12%。

嗯？

打开 ncu，看到了一个很抽象的事情：regs 涨到了 128，spill 非常非常多。


### 原因分析

代码大致是这样的：

```cpp
__device__ __forceinline__ void tcgen05_load_red_32dp32b32x_max_f32(
    uint32_t tmem_address,
    fp32_t& row_max
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile(
        "{\n\t"
        ".reg .b32 discard<32>;\n\t"               // ← 这里声明了 32 个虚拟寄存器
        "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 "
        "{discard0,discard1,...,discard31}, "     // ← 32 个输出，只有 max 有用
        "%0, [%1];\n\t"
        "}"
        : "=f"(row_max)
        : "r"(tmem_address)
        : "memory"
    );
#else
    row_max = -FLT_MAX;
    (void)tmem_address;
#endif
}
```

`tcgen05.ld.red` 的语法要求把 32 个输出全部列出来，但我们只需要 `max` 那一个值。其他 31 个输出是 discard 的。

问题在于：nvcc 并没有把 discard 寄存器优化掉。即使我们在代码里写了 `discard<32>` 并只用了 `discard0` 做输出，编译器仍然为 `discard0` 到 `discard31` 这 32 个值都分配了物理寄存器。

每条 `ld.red` 指令都占 32 个寄存器，再加上其他正在存活的变量，直接就把寄存器预算打满了。128 个寄存器触发 spill，性能自然崩了。

这确实很有意思——一条指令看似帮我们省了串行比较的代码，但代价是吃掉了大量的寄存器资源。在这个场景下，用 `ld.red` 还不如自己写串行比较。


### 小结

`tcgen05.ld.red` 是一个有用的指令，但它的设计适合那种“所有输出都有用”的场景（比如做 pooling 时取每个 tile 的 max）。对于 softmax 这种只需要一个值的场景，31 个 discard 的寄存器开销太大了，得不偿失。

这个方向暂时放弃，继续看其他优化点。

---

## V20: 精度上的取舍——一个让我很不舒服的版本

V19 的 `ld.red` 虽然翻车了，但说实话我有点舍不得放弃它。从语义上讲，用一条指令搞定整个 warp 的 max reduction 实在太优雅了，而且理论上它的吞吐应该比手写串行比较要高。

问题只是寄存器爆了。那换个思路：既然寄存器不够，就从别的地方省。

### 动了哪些地方

首当其冲的是那 32 个 accumulator 寄存器。PV 累加结果一直用 fp32 存着，能不能降成 fp16？光是降精度还不够——fp16 的动态范围可能撑不住整个序列的累加，而且寄存器位还是占着。

所以两个手段一起上：降成 fp16，同时搬到 smem 里去。虽然 smem 的读写延迟比寄存器高，精度也有损失，但如果能把 32 个寄存器彻底省出来，这个 tradeoff 可能还是划算的。

另外我还顺手动了几个地方：

- 用 `HFMA2` 替代部分 fp32 运算
- online softmax 的 `exp` 和 `sum` 也尝试用 fp16 做

说实话，这些改动在精度上是有风险的。fp16 做累加和做 exp，都有可能产生不可忽视的误差累积。但为了验证 `ld.red` 这条路能不能走通，我决定先跑一版看看。

### V20 的性能

```text
Elapsed Time (total):      0.0156 seconds
Average Time Per Step:     0.156 ms
Compute Performance:       220.52 TFLOPS
Memory Throughput:         401.12 GB/s
```

spill 确实都没了，性能回到了 220 TFLOPS 以上，甚至比 V16 的 204 还高了一点点。`ld.red` 的收益终于体现出来了。

### 但是……

这个版本在精度上不是我们想要的。

fp16 的 accumulator 和 fp16 的 softmax sum，虽然在当前的 benchmark case 上能跑通，但放在更长的序列或者更敏感的任务上，误差可能会被放大。用精度换性能这条路，从工程角度讲是有价值的，但作为一个通用 kernel，不能以牺牲数值稳定性为代价。

不过话说回来，也不是所有改动都不可取。softmax 的 sum 部分用 fp16 做，后续分母也是 fp16 的，这个精度风险相对可控，可以保留。但 accumulator 降精度这件事，我还是不太放心。

### 下一步：该搞流水线了

从 V14 到 V20，G4 的改动基本都在围绕 SM103 的新特性做适配——TMA、tcgen05、ld.red。性能从 V13 的 226 掉到 V14 的 179，又爬回到现在的 220，基本算是回到了 G3 末期的水平。

但有一个数字始终没变过：TC 利用率还是 7% 左右。

不管怎么折腾寄存器、怎么优化指令，TC 利用率就是上不去。这说明问题的根源不在计算侧，而在数据供给侧。当前的 pipeline 是串行的——TMA 加载 → tcgen05 计算 → softmax → PV → 下一轮。TMA 在加载下一块数据的时候，计算单元在空等；计算在跑的时候，DMA 在空等。

这才是真正的天花板。

V20 的 220 TFLOPS，在 SM103 的架构上只能算是一个地板——验证了指令集的正确性，验证了 ld.red 的可行性，但远没有发挥出硬件应有的潜力。

下一步的方向已经很清楚了：做流水线。把 TMA 的异步加载和 tcgen05 的计算重叠起来，让 DMA 和 tensor core 同时干活。

这应该是 G4 最后一个指令级别的优化了，再往下就是 pipeline 层的重构。

---

## G4 小结

G4 从 V14 到 V20，核心主题是从 SM80 迁移到 SM103，用上 TMA + TCGen05。V14 先把 QK 搬到了 tcgen05 上，V15 把 PV 也搬过去，V16 优化寄存器解决了 spill，V17 的 two pass PV 失败，V18 的 leader wait 也翻车，V19 的 ld.red 寄存器爆炸，V20 用精度换寄存器终于跑通了 ld.red，最终收在 220.52 TFLOPS。

从性能数字上看，G4 基本没涨——甚至比 G3 终点还略低一点。但这不代表 G4 是白干的，它验证了很多东西，也排除了一些错误方向。

### 几个比较明确的结论

1. SM103 的指令集确实有潜力，但需要正确的使用方式：tcgen05 一条指令顶 SM80 几十条指令，理论峰值很高。但它的编程模型更复杂，这一层抽象用好了是加速，用不好就是 overhead。

2. 寄存器还是最稀缺的资源：V19 的 ld.red 本质上是一条好指令，但 32 个 discard 寄存器直接让它变成了负优化。V20 通过把 accumulator 降精度并搬到 smem 才勉强省出寄存器空间，但在精度和访存延迟上都有妥协。

3. TC 利用率始终在 6-8% 之间徘徊：无论怎么改指令、怎么调寄存器，这个数字就是上不去。说明问题不在计算侧，而在数据供给侧——pipeline 是串行的，TMA 加载和 tcgen05 计算没有重叠。

4. G4 探索了很多方向，排除了不少错误答案：two pass PV、leader wait、ld.red 直接替换……这些方向在理论上都有吸引力，但实测都失败了。失败的尝试也是有用的记录。

### 下一步方向

G4 留下两个问题：TC 利用率为什么只有 7%？以及，一个 CTA 独占一个 SM，为什么 TMEM 和 smem（58 KB / 228 KB）都远没用满？

答案指向同一个地方：pipeline。现在的 mainloop 是"等 TMA → 算 QK → softmax → 等 PV → 读回 → 下一轮"的串行排队，任何时刻只有一个角色在干活，其他人全在等待。TMA、tcgen05、TMEM 这些硬件的验证工作 G4 已经做完了——接下来的问题不是它们好不好用，而是怎么把它们叠起来。

所以 G5 打算不再修修补补，直接重构整个 mainloop：

- Ping-pong 双缓冲：tcgen05 算当前 tile 的同时，TMA 异步加载下一个 tile，让 DMA 和 tensor core 同时干活
- 吃满独占 SM 的资源：TMEM 和 smem 的余量足以支撑更粗粒度的 stage 划分、更深的流水
- mbarrier 精细化管理：减少 warp 在同步上的空转


