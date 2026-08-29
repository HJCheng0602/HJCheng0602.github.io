---
title: "Flash Attention CUDA Kernel 优化: 从 56 到 986 TFLOPS on B300 — G5: Pipeline 重构与双缓冲 (221→727)"
date: 2026-07-31 00:00:00
description: "A CUDA kernel optimization note for Flash Attention on B300. G5: pipeline refactoring & double buffering."
series: flash-attention-b300
series_order: 5
timeline_title: "G5 · Pipeline 重构与双缓冲"
timeline_metric: "221 → 727"
tags:
    - CUDA
    - kernel optimization
    - flash attention
categories:
    - practice
---
![cover](cover.png)

## 背景


G4 从 V14 到 V20，核心主题是从 SM80 迁移到 SM103，用上 TMA + TCGen05。折腾了 7 个版本，最终收在 220.52 TFLOPS——基本没涨，甚至比 G3 终点还略低一点。

G4 结束时几个比较明确的结论是：

1. TC 利用率始终在 6-8% 之间徘徊：无论怎么改指令、怎么调寄存器，这个数字就是上不去。问题不在计算侧，在数据供给侧。
2. pipeline 是串行的：每个 tile 必须走完"等 TMA → 等 QK → softmax → sync → 等 PV → 读回 → sync"才能进下一轮，任何时刻只有一个角色在干活，其他人全在等待。
3. 寄存器还是最稀缺的资源：ld.red 是好指令但寄存器爆炸，精度换寄存器的方案有妥协。
4. G4 排除了不少错误答案：two pass PV、leader wait、ld.red 直接替换……这些方向理论上都有吸引力，但实测都失败了。

G4 没有带来性能突破，但它为 G5 铺了路——TMA、TCGen05、TMEM 这些东西都验证过了，下一步就是怎么把它们叠起来，让 DMA 和 tensor core 同时干活。

于是 G5 决定不再修修补补，直接重构整个 mainloop。


## V21: Ping-Pong Pipeline — 终于让 tensor core 和 TMA 同时干活了

G4 中我们似乎竭尽全力都没有超过 G3，但回头看，我们的 mainloop 始终是一个随便堆搭的状态——基本就是"等 TMA 加载完 → 算 QK → softmax → 等 PV 加载完 → 算 PV → 读回 → 下一轮"的串行流水线。

G4 终点（156.3 µs）的 ncu 数据很能说明问题：tensor core 利用率只有 7.8%，时间全花在等待上。任何时刻只有一个角色在干活，其他人全在摸鱼。

而且 G4 有一个设定：一个 CTA 独占一个 SM。但我们并没有最大化利用独占带来的资源——TMEM 只用了很小一部分，smem 也只用了 58 KB，还有很多余量。

V21 的目标就是：把 mainloop 改成 ping-pong 双缓冲流水线，让 TMA 加载和 tcgen05 计算重叠起来。

### 新的 Pipeline 设计

我们设计了如下所示的 mainloop（示意图），核心思路是两个 stage 交替：

![01](01.png)

让 TMA 在计算当前 tile 的时候，异步加载下一个 tile 的数据。这样 DMA 和 tensor core 就能同时干活了。

### Correction 操作

这里面有一个关键操作是 correction——也就是 PV 累加之后的 rescale 阶段。

之前的实现里，每个 warp 各自算 rescale 因子，然后各自读 TMEM、各自乘、各自写回去。这个模式的问题是：每个 thread 都要做一遍完整的 load + scale + store，寄存器占用高，而且 TMEM 的读写争抢严重。

V21 的 correction 核心函数是这样的：

```cpp
__device__ __forceinline__ void scale_output_stage(
    uint32_t tmem_base,
    fp32_t const* row_rescale,
    int32_t stage
) {
    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane_id = threadIdx.x & 31;
    int32_t row_local = (warp_id & 3) * 32 + lane_id;

    fp32_t history_scale = row_rescale[stage * UmmaM + row_local];

    #pragma unroll
    for (int32_t vector = 0; vector < HeadSize / 8; ++vector) {
        uint32_t values[8];
        ptx::tcgen05_load_32dp32b8x(tmem_base + vector * 8, values);
        ptx::tcgen05_wait_load();

        #pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            values[i] = __float_as_uint(__uint_as_float(values[i]) * history_scale);
        }

        ptx::tcgen05_store_32dp32b8x(tmem_base + vector * 8, values);
    }

    ptx::tcgen05_fence_before_thread_sync();
}
```

关键点在于：每一个 thread 负责一行。`row_local = (warp_id & 3) * 32 + lane_id` 这个映射保证了warp group中  128 个 thread 各自负责 Br=128 行中的一行，分工明确，没有竞争。

另外，我们让 warp 14 专门负责 barrier，warp 15 专门负责 epilogue 和 prologue。这样 barrier 和同步操作不再占用计算 warp 的资源。

### 这个设计带来了什么

1. Br 本质扩大了：虽然单个 UMMA 的 Br 不变（还是 64），但我们利用满了 TMEM 的容量，同时进行两个 stage 的 ping-pong。等效于在更粗的粒度上做 tile 管理，降低了 KV 的加载频率。

2. 专门的 rescale warp group：correction 操作现在由专门的 warp 来负责，不再跟计算 warp 抢寄存器。省下来的寄存器空间可以留给 tcgen05 的 fragment 和 TMA descriptor。

3. 精度完全符合预期：因为 rescale 因子还是 fp32 算的，只是 scale 操作本身用 fp32 做，没有精度损失——不像 V20 那样做了精度妥协。

### V21 的性能

跑了一下 benchmark：

```text
Elapsed Time (total):      0.0097 seconds
Average Time Per Step:     0.097 ms
Compute Performance:       355.80 TFLOPS
Memory Throughput:         647.20 GB/s
```

从 V20 的 220.52 TFLOPS 直接蹦到了 355.80 TFLOPS，提升了大概 61%。

这是 G1 V12 以来最大的一次单版提升。关键是方向对了——让 TMA 和 tensor core 同时干活，而不是排队等。

### 但还有问题

ncu 看了一眼，tensor core 利用率从 7.8% 涨到了 10% 左右——确实涨了，但还远没有到瓶颈。同时 spill 占比很高，说明寄存器压力还是大，虽然比 G4 好一些，但并没有完全解决。

下一步应该要优化 spill，把寄存器占用再往下压一压，让 occupancy 和 TC 利用率继续往上走。

---

## V22: Spill Free — 一个“理论上正确”的负优化

V21 虽然性能大涨，但 ncu 里 spill 占比还是很高。寄存器压力这个老问题又回来了——不过这次比 G4 好一些，至少有了明确的方向。

于是 V22 的目标很纯粹：把 spill 彻底干掉。

### 动了哪些地方

为了压缩寄存器占用，我试了几种手段：

1. 函数传参改为模板参数

之前很多函数传的是运行时变量，编译器没法做常量传播，寄存器分配也比较保守。改成模板参数之后，理论上编译器可以在编译期做更多优化，减少栈上临时变量的存活范围。

2. 指针计算移到 warp 分工的函数体内

上一版在 warp specialization 之前就把各种指针和地址算好了，导致这些变量在整条 pipeline 里都存活，占着寄存器不释放。现在把指针计算挪到真正使用它们的函数体内部，让它们的生命周期缩短。

3. 复用 values 数组

在 softmax 尾部有一个典型场景：load 出 8 个 fp32，然后每个算 exp，再转成 fp16，最后一条 STS.128 写出去。

这里的问题是：消费速度比供给速度快。load 出来的 values 在用完之后就死了，但实际上可以复用这些寄存器来存放转换后的结果，而不是另外开一套寄存器。

改成复用之后，理论上能省下不少寄存器。

### V22 的性能

```text
Elapsed Time (total):      0.0111 seconds
Average Time Per Step:     0.111 ms
Compute Performance:       310.25 TFLOPS
Memory Throughput:         564.34 GB/s
```

嗯，从 V21 的 355.80 TFLOPS 掉到了 310.25 TFLOPS，倒退了 13%。

但 spill 确实从一百万降到了十万——说明方向是对的，寄存器压力确实减轻了。

### 为什么变慢了

ncu 里看到一个不太妙的现象：stalled_no_instructions 上升了很多。

查阅资料之后，我猜测问题出在模板化引入了大量的不同地址的代码。也就是说，虽然寄存器省下来了，但编译器为了处理各种模板特化，生成了更多的指令和分支，导致 PC（程序计数器）相关的 stall 增加了。

spill 和 stalled_no_instructions 之间做了一个 tradeoff，而在这个 case 里，后者带来的损失超过了前者。简单来说：我们是减少了 spill，但代价是让 warp 更频繁地因为没指令可发而 stall，得不偿失。

### 小结

V22 的改动在直觉上都是对的——减少寄存器占用、缩短变量生命周期、复用寄存器空间。但它们组合在一起，却因为编译器行为的变化导致性能下降。这说明在 CUDA 优化里，局部最优的叠加不一定是全局最优。

下一步可能需要做更细致的消融实验，把这三个改动拆开来看，找出到底是哪一个在拖后腿。或者换一种思路——既然 spill 已经降到了十万，也许这个量级的 spill 本身不是瓶颈，真正的问题在别的地方。

---

## V23: Low Spill Compact — 验证归因，但问题还在

V22 去掉模板化之后，重新跑了一下 benchmark：

```text
Elapsed Time (total):      0.0096 seconds
Average Time Per Step:     0.096 ms
Compute Performance:       358.08 TFLOPS
Memory Throughput:         651.35 GB/s
```

性能回到了 V21 的水平，甚至略高了一点点（358 vs 355）。这说明上文的归因是正确的——模板化确实引入了大量的不同地址代码，导致 stalled_no_instructions 上升，抵消了寄存器优化带来的收益。

但代价是：spill 也回到了当时的水平。去模板化之后，spill 又回到了百万量级。V22 降低 spill 的目标没有达成，只是验证了“模板化是负优化”这个猜测。

所以现在的情况是：

- V21：355 TFLOPS，spill 一百万
- V22：310 TFLOPS，spill 十万
- V23：358 TFLOPS，spill 一百万

绕了一圈，回到了原点。

### 继续思考

现在看来，“降低 spill”和“保持性能”之间需要找到一个平衡点。模板化这条路走不通，但 spill 本身确实会影响 occupancy 和 TC 利用率——虽然 V21 的 358 TFLOPS 证明了这个量级的 spill 并不致命，但如果能降下来，应该还能再往前优化一点。

下一步的优化方向大概有这么几个：

1. 在非模板化的前提下，手动优化寄存器分配：不用模板参数，而是用 `__restrict__` 和 `const` 来帮助编译器做别名分析，或许能减少寄存器压力。
2. 调整代码结构，压缩变量存活范围：把大块代码拆成更小的函数，用 `__forceinline__` 控制内联程度，让编译器更早释放寄存器。
3. 接受当前 spill 水平，转向其他瓶颈：毕竟 358 TFLOPS 已经是当前最好的成绩，spill 也许不是最大的天花板——TC 利用率才 10%，这才是真正该关注的事。

---

## V24: Spill Free Loop — 终于找到了平衡点

V23 去掉模板化之后性能回到了 358 TFLOPS，但 spill 也回到了百万级别。V22 的模板化思路是对的——它确实降低了 spill——但代价是引入了过多的指令副本，导致 icache 压力和 stall 上升。

那我就在想：能不能保留模板化带来的寄存器优化收益，同时避免指令副本爆炸？

### 问题分析

回顾一下 V22 的模板化方案。softmax 的模板实例数是 2（两种配置），然后 key 循环展开因子是 16。所以总的指令副本数是：

```
2 × 16 = 32 份
```

而 V23（low spill compact）是：

```
1 × 16 = 16 份（但有 spill）
```

问题就出在这里：模板化本身只引入了 2 倍的代码膨胀，这不是问题。真正的问题在于 key 循环展开 16 次，让每份模板都膨胀了 16 倍，总共 32 份副本挤在 icache 里。

### 改进方案

V24 的改动思路很简单：模板照用，把 key 循环的展开因子从 16 砍成 1。

```cpp
#pragma unroll 1  // 之前是 #pragma unroll 16
for (int k = 0; k < HeadSize / UmmaK; ++k) {
    // softmax 核心计算
}
```

为什么可以这样？因为每一个 stage 后面都会接一个 `mbarrier`，同步开销已经把 stage 之间的边界卡死了。展开 16 次和展开 1 次，在端到端的延迟上没有区别——反正都要等 barrier。还不如不展开，省下 icache 空间，让寄存器优化得以保留。

这样一来，指令副本数变成了：

```
2 × 1 = 2 份
```

模板化带来的寄存器优化收益保留住了，代码膨胀问题也解决了。

### V24 的性能

跑了一下 benchmark：

```text
Elapsed Time (total):      0.0089 seconds
Average Time Per Step:     0.089 ms
Compute Performance:       384.99 TFLOPS
Memory Throughput:         700.29 GB/s
```

从 V23 的 358 TFLOPS 提升到了 384.99 TFLOPS，大概又涨了 7.5%。

更重要的是：现在 spill 被控制住了，指令副本也控制住了，编译器没有过度膨胀。ncu 里 stalled_no_instructions 也回到了正常水平。印证了先前的猜测——真正的问题不在模板化本身，而在模板 × 展开的组合爆炸。

终于找到了一个“既有寄存器收益、又没有性能回退”的平衡点。

---
## V25: 分工合作 — 拆掉那堵墙

V24 解决了 spill 和指令副本的矛盾，但还有一个问题让我一直不太舒服：warp 分工。

从上上图的时序图来看，上一版的 warp 分工虽然逻辑清晰，但执行下来感觉不怎么合理。具体来说是 Correction 的操作交给了专门的 warp group 来做，其他 warp 在干等。
上⼀版的时序图⼤概如下:
![02](02.png)

V25 的改动很简单：把 Correction 交还给 Wg0 和 Wg1 自己干，不再搞专门的 correction warp。

从直觉上讲，这似乎是“后退一步”——专门分工听起来更高效才对。但实测结果很有意思：

### V25 的性能

```text
Elapsed Time (total):      0.0085 seconds
Average Time Per Step:     0.085 ms
Compute Performance:       404.65 TFLOPS
Memory Throughput:         736.05 GB/s
```

从 V24 的 384.99 TFLOPS 提升到了 404.65 TFLOPS，大概又涨了 5.1%，突破了 400 TFLOPS。

ncu 里 spill 确实上升了，但先不管他——性能在涨，说明这个 tradeoff 是值得的。tensor core 利用率也变高了，虽然没有具体数字，但趋势是对的。

### 为什么专门分工反而不行

仔细想想，之前专门分工的设计虽然看起来“专业”，但实际上引入了一个问题：Correction 需要从 TMEM 里读数据、做 rescale、再写回去，这个过程本身是计算密集型的。把它交给专门的 warp 去做，相当于把一部分计算从主流水线上剥离出去了——但主流水线并没有因此跑得更快，因为它在等 Correction 的结果。

让 Wg0 和 Wg1 自己干 Correction，虽然打破了“分工明确”的漂亮设计，但减少了数据搬家和等待的额外开销，整体流水线更顺畅了。而且每个 warp 对自己的数据更熟悉，寄存器分配也更自然。

### 下一步

400 TFLOPS 是一个里程碑，但离 baseline 的 1156 TFLOPS 还有差距。tensor core 利用率虽然涨了，但应该还有上升空间。接下来看看剩下的几个 gap 在哪里——可能是 mbarrier 的 overhead，也可能是 TMA 的加载还没来得及做更细粒度的 overlap。

---
## V26: More Overlaps — 拆开 K/V 的 TMA 加载

V25 之后，ncu 显示 pipeline 中仍然存在一些空隙。观察时序图，发现在每个 tile 的起始阶段，K 和 V 的 TMA 加载是绑定的——同时触发、同时到达。但这两种数据的依赖关系并不相同：K 在 QK 之前就需要 ready，而 V 可以等到 softmax 之后。将它们绑定在一起，意味着其中一方 ready 后仍要等待另一方，造成了一个固定的等待窗口（上面时序图中标注的 gap C）。

### 改进思路

V26 把 K ready 和 V ready 解耦：

- K[t] 的加载只依赖 `qk_ready[t - 1]`，不再需要等 V。相比之前的实现，K 可以提前半个 tile 开始传输。
- QK[t] 的 gate 从 `KV_ready[t]` 改为 `k_ready[t] && correction[t - 1]`。
- softmax 的 gate 从 `qk_ready[t]` 改为 `qk_ready[t] && kv_free[t - 1]`，保证 TMEM 空间在 softmax 开始前可用。

新的信号图如下：

![03](03.png)

时序图：
![04](04.png)


在这个设计下，warp 13 的大部分搬运工作可以被隐藏：

- K transfer 藏在 QK done 之后的 softmax 和 PV 的间隙中
- V transfer 藏在 kv_free 之后的 QK 和 softmax 的间隙中

### V26 的性能

```text
Elapsed Time (total):      0.0067 seconds
Average Time Per Step:     0.067 ms
Compute Performance:       513.54 TFLOPS
Memory Throughput:         934.13 GB/s
```

相比 V25 的 404.65 TFLOPS，提升了约 27%，达到了 torch_sdpa 的 50%。

### 还存在的问题

ncu 显示 long scoreboard stall 有明显下降，说明解耦加载确实减少了等待。但同时出现了两个新的问题：

- spill read 上升：推测与额外的 mbarrier 同步有关
- no instruction：推测与 PV 迁移到 warp 15 发射后的调度空隙有关

这两项留到后续版本处理。

## V27: Overlap Compact — 压缩指令副本

V26 在解耦 K/V 加载后达到了 513 TFLOPS，但 ncu 显示 no instruction 仍然偏高，同时 spill read 也有上升。进一步看，问题出在循环展开上。

### 问题定位

w12 和 w15 的 key_tile 循环采用了全展开。具体来说，w12 每趟调用 `issue_qk_tile × 2`，每个内部是 8 次 `tcgen05_mma_1sm_f16`（各自带 elect.sync + setp + mma 的 inline asm）加 4 份描述符构造和推进；w15 类似，`issue_pv_tile × 2`，每个内部 2×8=16 次 MMA。内层循环本身是必要的发射流本体，但外层循环再乘以 8 展开后，发射流被复制了 8 遍——w12 一个 warp 独占几百条指令的直线代码，w15 同理。

这带来了两个问题：

- 取指压力：指令缓存被大量重复的展开代码占据
- spill：提前计算的长期值分散在展开副本中，增加了寄存器存活范围

### 改动

把外层循环从全展开改成 `#pragma unroll 1`，只保留内层循环的展开。

### 性能

```text
Elapsed Time (total):      0.0061 seconds
Average Time Per Step:     0.061 ms
Compute Performance:       559.52 TFLOPS
Memory Throughput:         1017.77 GB/s
```

相比 V26 的 513 TFLOPS，提升了约 9%。ncu 显示 no instruction 下降了 72%，spill read 下降了 18%，符合预期。long scoreboard 仍然偏高，留到下一版处理。
---
## V28: More Precise Barriers — 拆分 Correction Barrier

V27 之后，long scoreboard 仍然偏高。重新审视 correction barrier 的语义，发现它实际上承载了两层含义：

1. softmax 完成（w12 需要等这个才能覆写 QK 的 TMEM）
2. rescale PV TMEM 完成（w15 需要等这个才能开始下一轮 PV）

但 w12 只需要等待 softmax 完成，不需要等待 rescale。原先用同一个 barrier 会导致 w12 多等一些不必要的时间。于是引入了一个新的 barrier（score free barrier），专供 w12 使用。

### 性能

```text
Elapsed Time (total):      0.0059 seconds
Average Time Per Step:     0.059 ms
Compute Performance:       578.53 TFLOPS
Memory Throughput:         1052.34 GB/s
```

相比 V27 的 559 TFLOPS，略有提升，但改动本身不大。

神的目光开始聚焦 softmax。

---
## V29: Softmax 16x — 调整 TMEM 加载位宽

在 softmax 内部，`tcgen05.ld` 的位宽设置之前统一为 8。经过测试，将 softmax 内部的 load 位宽改为 16，其余部分保持不变。

```text
Elapsed Time (total):      0.0058 seconds
Average Time Per Step:     0.058 ms
Compute Performance:       594.22 TFLOPS
Memory Throughput:         1080.88 GB/s
```

相比 V28 的 578.53 TFLOPS，提升了约 2.7%。`tcgen05.ld` 位宽的调整对 softmax 阶段有一定收益，但尚未触及更深层的瓶颈。

---
## V30: Barrier 细化

稳态关键路径原为 `softmax0[t] → cd[0] → PV0[t] → PV1[t] → softmax0[t+1]`。但 PV1 的结果在下一轮 softmax 开始前并不需要，softmax0 实际上只需要等待 PV0 完成即可。

引入 PV0 barrier 后，softmax0 不再等待 PV1，信号图据此调整。

![05](05.png)

```text
Average Time Per Step: 0.053 ms
Compute Performance: 643.07 TFLOPS
Memory Throughput: 1169.75 GB/s
```

相比 V29 的 594 TFLOPS，提升约 8.3%。softmax0 不再等待 PV1，这个 gap 被消除。目前 profile 中已经看不到显著的 stall 项，再往下做可能就需要从全局数据流或数值精度角度重新审视了。
---
## V31: Alternate Stage Priority — 尝试动态切换

V30 引入 PV0 barrier 之后，稳态关键路径缩短了约 8%。顺着这个思路，理论上可以让 PV0 和 PV1 的角色在奇偶 tile 之间交替，使得两个 stage 的负载更均衡，进一步压缩关键路径。

于是按奇偶 tile 分别配置不同的 first 对象，期望在保持 ping-pong 吞吐的同时，让两个 PV stage 轮流优先完成。

```text
Average Time Per Step: 0.053 ms
Compute Performance: 643.07 TFLOPS
Memory Throughput: 1169.75 GB/s
```

性能与 V30 持平。spill 有所下降，但不在关键路径上。动态切换引入了额外的分支和 barrier 配置开销，long scoreboard 略有上升。权衡之下，收益不明显，回退此改动。
---
## V32: Epilogue 优化 — 尝试用 TMA 做写回

V31 之后，性能稳定在 0.053 ms 左右。本轮把注意力转向 epilogue——从 TMEM 读回结果并写回 global memory 的阶段。

原先的 epilogue 实现是每个 thread 负责一行，直接从 TMEM 读，然后写回 gmem。但每一行之间相距 16 KB，访存 pattern 毫无 coalescing 可言，直觉上应该还有优化空间。

### 改动

尝试用 TMA 来替代原有的 epilogue 写回路径：

1. `tcgen05.ld` 从 TMEM 读到寄存器
2. 每行除以 sum，转 FP16
3. 写回 shared memory（复用原有的 smem_P 空间，不新增 smem）
4. TMA engine 发起 2 笔 bulk store 写回 gmem

这样理论上可以利用 TMA 的硬件 coalescing 能力，把原本分散的标量写合并成批量 DMA 传输。

### 性能

```text
Elapsed Time (total):      0.0059 seconds
Average Time Per Step:     0.059 ms
Compute Performance:       578.02 TFLOPS
Memory Throughput:         1051.41 GB/s
```

从 0.053 ms 回退到了 0.059 ms。ncu 分析下来，TMA 引入了额外的同步开销，而且 epilogue 本来就不在关键路径上，改动的收益被同步开销抵消了。回退此改动。

---
## V33: log2exp2 — 压缩 softmax 计算链路

V32 回退后重新审视 softmax 的计算路径，发现之前一直有一个 3 指令的序列可以用 2 条指令完成。

### 改动

旧链路：

```
FFMA(x · Scale − max) → FMUL(×log₂e) → MUFU.EX2
```

新链路：

```
FFMA(x · ScaleLog2 − max₂) → MUFU.EX2
```

把 log₂e 的乘法提前合并到 scale 里，省掉了一条 FMUL。

### 性能

```text
Elapsed Time (total):      0.0047 seconds
Average Time Per Step:     0.047 ms
Compute Performance:       724.16 TFLOPS
Memory Throughput:         1317.24 GB/s
```

从 V32 的 0.053 ms 降到了 0.047 ms，提升了约 13%。

这个优化在 G2 的 V6 就试过，但当时整个 softmax 不在关键路径上，指令数的压缩没有转化为端到端的收益。现在 softmax 处于主路径上，long scoreboard 占比上升是预期的——计算链路压缩后，访存等待的比例自然变高了，但这说明优化已经打到了正确的位置。
---
## V34: Split V — 尝试压缩 V 的存活窗口

V33 压缩了 softmax 计算链路后，性能到了 0.047 ms。此时重新审视 PV 阶段的访存模式，发现 V 的存活窗口可能还可以进一步压缩。

### 问题分析

PV 总共涉及 4 个“四分之一块”（2 个 P × 2 个 V 半块）。旧顺序按 P 分组：P0×V0、P0×V1、P1×V0、P1×V1。V0 要一直活到第 3 个四分块（P1×V0）结束才没人读。

如果改成按 V 分组：P0×V0、P1×V0、P0×V1、P1×V1，V0 在第 2 块结束就可以被覆写，比旧版提前了一半。于是：

- w13 搬下一趟 V0 的发射点从 `kv_free[t]` 提前到 `v0_free[t]`
- 首个 PV 四分块只需等 16KB 的 V0 到货，不用等全部 32KB——TMA 到货延迟对关键路径的贡献减半

### 性能

```text
Elapsed Time (total):      0.0053 seconds
Average Time Per Step:     0.053 ms
Compute Performance:       643.76 TFLOPS
Memory Throughput:         1171.00 GB/s
```
北京大学大败！清华大学大胜！
从 V33 的 0.047 ms 回退到了 0.053 ms。

### 分析

ncu 显示按 V 分组后，P0 free 要等到 mma 发射之后才能结束，主路径被拖慢了。softmax 仍然占大头，这次改动实际上延长了 softmax 到 PV 之间的等待窗口，回退。
---
## V35: rescale ld/store 位宽调整

V33 的 log2exp2 把 softmax 计算链路压缩到 2 条指令后，性能到了 0.047 ms / 724 TFLOPS。此时 profile 里 rescale 阶段的 ld/store 占比开始变得显眼。尝试将 rescale 的 load/store 位宽从 8 调整到 16，减少 TMEM 访问的指令数。

```text
Elapsed Time (total):      0.0047 seconds
Average Time Per Step:     0.047 ms
Compute Performance:       727.25 TFLOPS
Memory Throughput:         1322.85 GB/s
```

相比 V33 的 724 TFLOPS，略有提升，约 0.4%。不算大，但属于无成本的优化。

---

## V36: Q prescale — 理论正确但实际翻车

V35 之后，我注意到一个点：softmax 里每次都要做 `FFMA(x * ScaleLog2 - max)`，其中的乘法是否可以在 Q 加载阶段就做完？如果把 scale 提前乘到 Q 上，那么 softmax 内部就变成 `exp2(x - max)`，省掉一条乘加指令。

想法很直接，实现也简单：在 TMA 加载 Q 之后做一次 prescale，让 Q 本身变成 scaled 版本，后续所有用到 Q 的地方都直接用这个版本。

预期收益是 softmax 的 ffma 变成 fadd，省一条指令。但跑出来是：

```text
Elapsed Time (total):      0.0049 seconds
Average Time Per Step:     0.049 ms
Compute Performance:       696.25 TFLOPS
Memory Throughput:         1266.48 GB/s
```

从 V35 的 727 TFLOPS 掉到了 696 TFLOPS。

查阅资料之后发现原因：原先的 `exp2(x * ScaleLog2 - max)` 是一条 FFMA + EX2，编译器会融合成单条 FMA 指令；而 `exp2(x - max)` 是 FADD + EX2，看似少了一次乘法，但那个乘法本来就在 FFMA 里被融合了，并没有额外的指令开销。反倒是因为做了 prescale，每次 Q 加载时都要多算一次乘法，额外增加了操作。
---
## G5 小结

G5 从 V21 到 V36，核心主题是 pipeline 重构 + 双缓冲 + 精细同步。

V21 首次引入 ping-pong 流水线，性能从 220 TFLOPS 直接跃升至 355 TFLOPS。后续版本围绕指令副本、分工、barrier 粒度、softmax 计算链路做了一系列迭代。V33 的 log2exp2 优化将 softmax 计算链路从 3 条指令压缩到 2 条，性能进一步提升。V35 调整 rescale ld/store 位宽后，最终稳定在 727.25 TFLOPS。

### 性能演进回顾

| 版本 | 核心改动 | Time/step | TFLOPS |
|------|---------|-----------|--------|
| V20（G4 终点） | G4 收尾 | 0.156 ms | 220.52 |
| V21 | Ping-pong 流水线 | 0.097 ms | 355.80 |
| V22 | Spill free 尝试（模板化） | 0.111 ms | 310.25 |
| V23 | 去掉模板化 | 0.096 ms | 358.08 |
| V24 | 循环展开 16→1 | 0.089 ms | 384.99 |
| V25 | Correction 交还 Wg0/Wg1 | 0.085 ms | 404.65 |
| V26 | 解耦 K/V TMA | 0.067 ms | 513.54 |
| V27 | 外层循环展开 8→1 | 0.061 ms | 559.52 |
| V28 | 拆分 correction barrier | 0.059 ms | 578.53 |
| V29 | softmax load 位宽 16 | 0.058 ms | 594.22 |
| V30 | 引入 PV0 barrier | 0.053 ms | 643.07 |
| V31 | 奇偶 tile 动态切换 | 0.053 ms | 643.07 |
| V32 | TMA epilogue | 0.059 ms | 578.02 |
| V33 | log2exp2 | 0.047 ms | 724.16 |
| V34 | 按 V 分组 PV | 0.053 ms | 643.76 |
| V35 | rescale ld/store 位宽 16 | 0.047 ms | 727.25 |
| V36 | Q prescale | 0.049 ms | 696.25 |

### 几个比较明确的结论

1. Pipeline 重构是 G5 最大的收益来源：V21 首次将 TMA 加载和计算重叠，从 220 直接跳到 355 TFLOPS，说明之前的主要瓶颈在串行等待，不在计算本身。

2. 指令副本膨胀比预期更严重：V22 和 V24 的对比说明，模板化本身不是问题，但模板 × 循环展开的组合会导致指令缓存压力。把外层展开从 16 或 8 降到 1，在保留寄存器收益的同时消除了 no instruction stall。

3. Overlap 的边际收益在递减：V26 解耦 K/V TMA 带来了 27% 的提升，但后续 V30 的 barrier 细化、V33 的 log2exp2 都只有个位数百分比。越往后，能藏的东西越少。

4. 一些优化在 G2 无效，但在 G5 有效：log2exp2 在 G2 的 V6 就试过，当时几乎没有收益。现在 softmax 处于主路径上，压缩计算链路带来了 13% 的提升。这说明优化的上下文比优化本身更重要——瓶颈位置变了，同样的优化效果也不同。

5. TC 利用率从 7.8% 涨到了约 18-20%：虽然没有达到理想的饱和状态，但已经说明 pipeline 重构方向是正确的。

### 下一步方向

G5 收在 727.25 TFLOPS，大约是 torch_sdpa 的 63%。但 ncu 里有一个数字越来越刺眼：TC 利用率停在 18-20% 上不去了。

从 V26 到 V35，我们做的事本质上是同一件：把流水线里的空隙一个个找出来填掉——解耦 K/V、拆 barrier、压缩 softmax 计算链。每一版都有收益，但收益从 27% 一路掉到 0.4%。这让我隐约有个感觉：单 CTA 的资源可能已经用得差不多了。一个 CTA 能调度的 warp 数就那么多，无论 barrier 怎么精细、overlap 怎么排布，tensor core 就是喂不满。

于是下一步想试试一个之前没碰过的方向：引入 cluster，用 2 个 CTA 协同工作。多一个 CTA，就多一套 warp、多一份并发度，也许能打破单 CTA 的天花板。
