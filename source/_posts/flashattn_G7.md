---
title: "Flash Attention CUDA Kernel 优化: 从 56 到 986 TFLOPS on B300 — G7: 回归 1CTA (880→987)"
date: 2026-07-31 00:00:00
description: "A CUDA kernel optimization note for Flash Attention on B300. G7: back to 1CTA."
series: flash-attention-b300
series_order: 7
timeline_title: "G7 · 回归 1CTA"
timeline_metric: "880 → 987"
tags:
    - CUDA
    - kernel optimization
    - flash attention
categories:
    - practice
---
![cover](cover.png)

## 背景


G6 从 V37 到 V44，尝试了 2CTA cluster，最终收在 880.14 TFLOPS。V44 之后尝试了多版优化都没能突破 0.039 ms。用 ncu 逆向看了 torch sdpa 的实现，发现它用的是 1CTA。这说明在 SM103 上 1CTA 的资源并没有到天花板，搞 2CTA 反而多了一套 cluster 同步的开销。

于是 G7 决定回到 1CTA，把 G6 验证有效的技术（conditional scale、PV 细分、手写 barrier）迁移回来，看看在更简单的同步模型下能跑到多少。


## V45: 百废具兴

回到 1CTA。KV 读双倍的收益消失了，信号图大概变成这样：

![01](01.png)

PV 和 QK 是分开发的，某种意义上自动完成了 interleave。KV 保留了 6 环（K、V 各 3 级交错），环数和 2CTA 一致；少了一半的是 2CTA 时代 cluster 内部 KV 数据复用的收益——回到 1CTA 后，每个 CTA 都得自己读一遍 KV。

### 性能

```text
Elapsed Time (total):      0.0043 seconds
Average Time Per Step:     0.043 ms
Compute Performance:       797.73 TFLOPS
Memory Throughput:         1451.07 GB/s
```

从 G6 终点的 880 掉到了 797。barrier 确实少了很多，ncu 看 KV 减半似乎也没带来什么负面影响——DRAM 读没变，推测 L2 cache 起到了不错的缓冲作用。

性能回退的原因大概有两个：一是 w8 覆写 S 的 wait 条件从"softmax 做完"退化成了"PV 做完"，多等了一段；二是 persistence 消失了，有一部分 CTA 被销毁后重建，可能也会影响一些。

不过这一版没有通过正确性检查。(想不到吧（x)

---
## V46: 拨乱反正

上一版没通过正确性检查，这一段时间全部花在把它修回来。前后大约二十个版本，绝大多数是探针而非优化。

QK 和 softmax 一直都是对的，问题似乎在 V 的软件重排公式上，两个 XOR 项各错了一个地方。

背景大概是这样的。1CTA 的 PV 用 TS 形式，A 是 TMEM 里的 P，B 是 smem 里的 V。V 需要摆成 MN-major 加 SW128 swizzle 的样子，而 TMA 只会按行主序搬，所以只能由发射 warp 用软件把 V 重排到一块 scratch 里。为此把 KV 环从 6 级降到 4 级，腾出两个环（32KB）。重排时每个元素的目标地址是：

```
offset = atom基址 + physical_k_row * 64 + swizzled_n_vector * 8
```

错的是这两项：`swizzled_n_vector` 写成了 `n_vec ^ (k & 6)`，只用了 k 的高位，导致奇偶 k 的 N 向量整体错开一位；`physical_k_row` 写成了 `k ^ n_atom`，让第二个 N64 块多做了一次 K 行交换。第一处影响所有列，第二处只影响后 64 列。所以症状是部分对部分错，max 0.74、mean 0.054，既不是 NaN 也不是全错。

改法就是两行：

```cpp
physical_k_row = k_in_atom;
swizzled_n_vector = n_vector_in_atom ^ k_in_atom;
```

改完就好了。

过程比结论麻烦多了。前期一直在猜，猜的方向基本都错：怀疑过双发射 warp 的 TCGEN commit 竞态，合回单发射后误差一字不变（不过 single_issue 这个结构留下来了）；怀疑过 TS-A 的地址要乘 2，这个假设在 2CTA 时期就失败过一次，这次又失败一次，直接就 NaN 了；怀疑过 1CTA 的 TS 指令少写了四个 disable_mask 操作数，补上以后没有影响；还怀疑过 P 要用 `tcgen05.st...unpack::16b` 展开着写，为此把 TMEM 的列规划整个推倒重来，绕了三个版本，最后被证明 TS-A 要的就是密集 half2，和 2CTA 完全一样。

转折点是发现两个不同的 V 重排公式产出了完全相同的错误值。这说明继续改公式没有任何信息增益，得换测量方法。于是改成基向量探针：把 V 临时设成单位向量，只有 V[k][0]=1，其余全零，这时 PV 的输出理论上必须精确等于 P[k]。然后扫描全部 128 个输出列，看非零峰值落在哪、数值最接近哪个 P[k]。现在运行就能反推出硬件实际使用的映射表。

结果一目了然。注入 (k=1, n=0)，峰值出现在 n=8，说明 N 向量翻转了一位；注入 (k, n=64)，硬件读成 (k^1, n=72)，说明第二个 atom 被多交换了一次 K 行。两个公式错误各自出现。

因此得知：库里的布局数字不能直接搬进裸 PTX。CuTe 的宽度换算、typed fragment 的 row stride、scalar layout 的 k ^ n_atom，三次移植三次失败。库的中间表示和指令的物理语义之间隔着它自己的封装层。

探针本身也有盲区。中途一度以为 TMEM 的 row 32/64/96 读出重复值是个 bug，其实是发射 warp 无法跨 TMEM subpartition 读取造成的假象，正式的四个 epilogue warp 各读自己的 datapath，并没有问题。

这一段没有产生任何性能收益，KV 环还从 6 级降到了 4 级，V 多了一次软件重排的开销。但至少现在拿到了一个正确的 1CTA 基础，后面的数字有意义。

额之所以现在才遇到，是之前一直绕过了 N col 128 的问题，现在尝试去做这件事情。因此这算是第一次出错？

```text
Elapsed Time (total):      0.0065 seconds
Average Time Per Step:     0.065 ms
Compute Performance:       530.46 TFLOPS
Memory Throughput:         964.89 GB/s
```

认为软件 V swizzle 占用了大量的时间。
---
## V47: 5D TMA

上一版软件 V swizzle 直接把性能从 797 干到了 530。V 重排的那一趟 smem 往返串行地插在发射 warp 的关键路径上，环深从 6 掉到 4，时间全花在那里了。

我们似乎可以用一个超级抽象的 TMA 来替代这件事。

具体来说，不用软件做 V 的 MN-major + SW128 重排了，直接让 TMA 在搬运的时候完成这个布局转换。TMA 的 5D tensor descriptor 可以描述任意的维度排列和 swizzle，只要把 global memory 里的 V 映射到 smem 里 tcgen05 TS 期望的 layout 就行。

具体的 descriptor 设计踩过一次坑。先看失败的 4D 版本（pv_n128 时代）：

```
dimensions = {64, 2, heads, seq}      box = {64, 2, 1, 128}
遍历顺序(内→外): 64特征 → 2原子 → 128行
写出的 smem: 行0[原子A][原子B] 行1[原子A][原子B] ...   ← 两原子并排，错
```

两个原子并排写在每一行里，和 tcgen05 TS 期望的"原子块序"正好相反。

这版的 5D：

```
dimensions = {64, 8, 2, heads, seq/8}  box = {64, 8, 2, 1, 16}
strides    = {行距, 128B, 256B, 8×行距}
遍历顺序(内→外): 64特征 → 8个key → 2原子 → head → 16个key组
写出的 smem: [原子A的完整 8×64块][原子B的完整 8×64块] 下一个key组 ...   ← 原子块序，对
```

秘诀是在描述符里显式加一维来枚举原子：dim1=8 沿 key 走（stride = 整行）、dim2=2 沿特征走 64 个元素（stride=128B）。这样把嵌套顺序从"行内跨原子"翻成"原子内跨行"。展开这个遍历序得到的地址正是：

```
offset = k_group*1024 + n_atom*512 + k_in*64 + n_vec*8
```

这样发射 warp 就不用碰 V 的数据了——只需要配好 descriptor，TMA 硬件自己会按照指定的 layout 把 V 搬到 smem 里。

看起来非常合理。

```text
Elapsed Time (total):      0.0039 seconds
Average Time Per Step:     0.039 ms
Compute Performance:       884.19 TFLOPS
Memory Throughput:         1608.33 GB/s
```

从发射 warp 的关键路径上删掉了一次串行的 64KB smem 往返，同时环深恢复到了 6 级。而且没有牺牲任何东西。
---
## V48: Dual Issue?

尝试让发射 warp 双发指令，期望能进一步压榨 TC 的吞吐。

```text
Elapsed Time (total):      0.0043 seconds
Average Time Per Step:     0.043 ms
Compute Performance:       797.68 TFLOPS
Memory Throughput:         1450.97 GB/s
```

拉完了。看一眼 ncu，barrier 增加是 dual issue 的代价，得不偿失。
---
## V49: Split P 64

尝试把 V43 的 96+32 改成 64+64，看看对称拆分会不会有什么变化。

```text
Elapsed Time (total):      0.0039 seconds
Average Time Per Step:     0.039 ms
Compute Performance:       883.29 TFLOPS
Memory Throughput:         1606.69 GB/s
```

看上去没有提升也没有下降。看看 ncu 也没什么太大区别。摆烂了。

看看能不能优化优化 SSC。
---
## V50: Softmax Pipeline 双缓冲

做了两处改动。

改动一：max 归约的 wait 从 4 次并成 1 次

旧：

```cpp
for chunk = 0..3 {
    ld.red(chunk) → chunk_max;
    wait_load();  // 每轮都等
}
tile_max = max(...);
```

新：

```cpp
fp32_t chunk_max[4];
for chunk = 0..3 {
    ld.red(chunk) → chunk_max[chunk];
}
wait_load();  // 只等一次
tile_max = max(...);
```

4 次 TMEM 往返从串行变成并发，只暴露一次延迟。代价是 4 个 fp32 寄存器。

改动二：转换循环的 score 装载改双缓冲

旧：

```cpp
for v = 7..0 {
    load(v) → scores;
    wait;  // 裸露
    scores 转换 → packed;
    store P;
}
```

新：

```cpp
load(7) → scores_current;  // 序幕
wait;

for v = 7..0 {
    if (v > 0) load(v-1) → scores_next;  // 先发出去
    scores_current 转换 → packed;
    store P;
    if (v > 0) {
        wait;
        scores_current = scores_next;
    }
}
```

新增 `scores_next[16]`（16 个寄存器）和末尾的 16 条搬移。

每个 tile 每个 softmax 线程原来有 12 次串行 TMEM 往返（4 + 8），现在只剩 1 次裸露、11 次被计算掩盖。

```text
Elapsed Time (total):      0.0037 seconds
Average Time Per Step:     0.037 ms
Compute Performance:       929.86 TFLOPS
Memory Throughput:         1691.41 GB/s
```

有提升！SSC 确实下降了。
---
## V51: Persistent CTA

引入 G6 的 persistent CTA 思路，拒绝一个 CTA 内反复 init，只分配 task。

```text
Elapsed Time (total):      0.0035 seconds
Average Time Per Step:     0.035 ms
Compute Performance:       986.76 TFLOPS
Memory Throughput:         1794.90 GB/s
```

不错，但是我燃尽了。就这样吧，已经达到了 86%，够了，该去看看其他的了。


## G7 小结

放一个终版的 baseline 对比：

```
kernels/tests/ops/attn/flash_attn_jincheng_test.py
S=1024 H=64 D=128 fp16 non-causal, 34.4 GFLOP per call

implementation              ms      TFLOPS    SNR       err vs ours
-------------------------------------------------------------------
sdpa (auto)                 0.0295  1163.8    1.05e-07  0.85x
sdpa cudnn                  0.0321  1070.5    1.05e-07  0.92x
jincheng_1 (ours)           0.0349  984.7     1.85e-07  1.00x
flashinfer prefill          0.1027  334.6     1.03e-07  2.94x
sdpa flash                  0.1056  325.3     1.03e-07  3.03x
sdpa mem_efficient          0.2112  162.7     1.03e-07  6.05x
torch unfused               0.2196  156.5     1.14e-07  6.29x
jincheng_0                  1.2412  27.7      1.03e-07  35.57x
sdpa math                   1.2675  27.1      3.30e-10  36.32x
flash-attn (Dao)            skipped (unavailable)
```

G7 从 V45 的 797 起步，经过 V46 拨乱反正（掉到 530）、V47 的 5D TMA（回到 884）、V48 的 dual issue 翻车（回 797）、V49 的 split P 64 无变化、V50 的 softmax pipeline 双缓冲（冲到 929）、V51 的 persistent CTA（到 986），最终收在 986.76 TFLOPS。从 torch sdpa 的 1163.8 来看，达到了约 85%。

### 整个系列

7 篇博客，51 个版本，从 V1 的 56.83 TFLOPS 到 V51 的 986.76 TFLOPS，17.4 倍。对照 torch sdpa（auto）的 1163.8 TFLOPS 是 85%，对照 sdpa cudnn 的 1070.5 是 92%——而 flashinfer prefill、sdpa flash、mem_efficient、torch unfused 这些实现，全都在我们后面。

回看整条路，大概可以分成三个阶段：

| 阶段 | 系列 | 主题 | TFLOPS |
|------|------|------|--------|
| SM80 风格 | G1–G3 | bank conflict → softmax 串行链 → fragment reshape | 56.83 → 226.83 |
| 迁移 SM103 | G4 | TMA + TCGen05，验证指令集 | 226.83 → 220.52 |
| 流水线 | G5–G7 | ping-pong → cluster → 回归 1CTA | 220.52 → 986.76 |

### 一些教训

1. 优化要打在关键路径上（G2 V6）：在旁支上做 FMA 替换，就像给堵在高速上的跑车换更好的火花塞——车还是动不了。
2. ncu 的 Speedup 估算有前提：Local Speedup 假设 compute 是瓶颈。kernel 是 latency-bound 时，那 47% 就是幻觉。
3. 寄存器比 shared memory 更稀缺：V9 的双缓冲、V19 的 ld.red 都栽在寄存器上。在 SM103 上，寄存器压力往往比 smem 容量更早成为天花板。
4. 架构级改动的收益远大于微调：G2 六个版本涨 11%，V12 一个改动涨 52%，V21 一个改动涨 61%。
5. 瓶颈在移动："谁在等谁"比最终数字更有信息量。每次瓶颈的移动，都对应着上一轮优化的成功——SHFL → LDS → stmatrix → mma → mbarrier，一路搬了四次家。
6. 优化的上下文比优化本身更重要：log2exp2 在 G2 是零收益，在 G5 是 13%。瓶颈位置变了，同样的优化效果完全不同。
7. 失败的尝试是必要的：two pass PV、leader wait、ld.red、dual issue、Q prescale、模板化……它们各自排除了一个错误答案。失败也是记录。

### 结束

写这个系列最大的感受是：优化 kernel 不是堆技巧，而是建立"瓶颈在哪"的直觉。56 到 986 的距离，不是某一个魔法优化拉开的，而是 51 个版本里每一次"瓶颈移动到下一个位置"的累积。

最后留下的问题也清楚：softmax 还占着一部分时间，TC 利用率离饱和还有距离，1CTA 能跑赢 2CTA 说明资源还没用完。986 不是终点，只是这一轮的收尾。

就这样吧，我力竭了。
