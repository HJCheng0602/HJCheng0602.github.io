---
title: "TopK kernel 优化: from 20GB/s to 2024GB/s on B300"
date: 2026-07-07 00:00:00
description: "A CUDA kernel optimization note for TopK selection on B300."
tags:
    - CUDA
    - kernel optimization
    - topk
categories:
    - practice
---

## Introduction

上上周五写了 topk kernel 的第一版，应该是 index 错误导致我查了一下午 bug，想要提前写完一些 kernel 然后去美美参加 Optiver FutureFocus 的心情荡然无存。本周回来之后，本来不打算继续写 topk kernel 了，但 mt 还是建议我去写一下 topk kernel，因为 topk 包含了一些很重要的知识点，于是我便开始着手优化该 kernel，其中版本迭代和性能提升的过程也很有代表性，我决定写一篇博客记录一下 topk kernel 的优化过程。

这次优化的目标场景还是比较固定的：
```
input:  一维 fp32 tensor, shape = [N]
N:      32M ~ 256M
K:      32
output: 只返回 topk values
order:  descending
GPU:    NVIDIA B300
Torch:  2.11.0 + CUDA 13.0
```

这个场景和 PyTorch torch.topk 并不完全相同。PyTorch 的 topk 是一个通用算子，需要支持多维 tensor、不同 dtype、任意 K、返回 values 和 indices，以及更多边界情况。而这里是一个高度特化的 kernel：一维、fp32、K=32、values-only。因此本文中的对比主要是为了分析优化过程，而不是说这个 kernel 可以直接替代通用 topk。

最后的性能从最早大约 50GB/s，一路优化到：
```
N = 268,435,456
time ≈ 0.526 ms
effective input bandwidth ≈ 2.0 TB/s
```

整个过程经历了多个版本：merge insert、local bitonic sort、coalesced load、shared memory padding、merge 函数改写、radix select 尝试、multi-block stage2 reduction，以及最后的 SoA layout 实验。每一次明显的性能变化背后都对应一个具体瓶颈的解决，而不是简单地“多加几个 pragma”产生的free fruit。

## Overall Algorithm

TopK 的基本目标是从 N 个数里找出最大的 K 个数。在 N 很大、K=32 很小的情况下，完整排序整个 input 显然是不划算的。我们只需要前 32 个值，不关心剩余元素的相对顺序。

因此整个 kernel 自始至终便采用两阶段结构：
```
stage1:
  每个 block 处理 input 的一个 tile
  每个 block 输出自己的 top32

stage2:
  合并所有 block 的 top32
  得到全局 top32
```
也就是：
```
input[N]
  ↓
partial_topk[num_blocks, 32]
  ↓
global_topk[32]
```
这个主要的架构一直都没有改变。本文所做的优化主要集中在以下几个问题：

1. 每个 thread 如何处理自己的元素？
2. 每个 block 如何高效合并 thread-local topk？
3. stage2 如何合并所有 block 的 partial topk？
4. shared memory layout ？
5. 哪些部分是瓶颈，哪些优化其实是负优化？

## V1: First Version — Merge Insert Based TopK

v1 版本的 topk kernel 是以 merge insert 为基础，主要分为两个阶段。

在 stage1 中，每个 thread 读取若干个 input 元素，然后维护一个长度为 K 的 local topk 数组。每读到一个新元素，就判断它是否应该插入当前 topk。如果这个值比当前 topk 里的某些值更大，就把它插入到合适的位置，并把后面的元素往后移动。

伪代码是：

```
float local_topk[K];

for each value v assigned to this thread:
    insert v into local_topk if v is large enough
```

当每个 thread 都得到自己的 local topk 后，block 内再做 reduction merge。两个已经降序排列的 topk 数组可以用类似 merge sort 的方式合并，只保留前 K 个值：

```
left_topk[32]
right_topk[32]
→ merged_topk[32]
```

一个 block 内有 256 个 thread，因此需要多轮 merge，最终每个 block 输出一个 top32：

```
block input tile
  ↓
block top32
```

stage2 再把每个 block 的 top32 合并，得到全局 top32。

这个版本的性能大约只有 50GB/s，显然很差。最主要的问题是 thread 内的 insert topk 有一点naive。每个元素插入 topk 数组时，最多要比较和移动 32 次，而且移动位置是 data-dependent 的，无法unroll。这类代码在 CPU 上比较自然，但在 GPU 上不太友好，因为它会产生较多分支、predicate 和串行依赖。

除此之外，早期版本的访存也不够理想。每个 thread 如果读取自己连续的一段数据，例如：

```
input[tid * VPT + i]
```

这个 load pattern 是 thread-contiguous，而不是 warp-contiguous。每个 thread 自己读连续的 VPT 个元素，生成的 ldg128，单线程访存并不差；但从单条 warp load instruction 的角度看，相邻 lane 访问间隔为 VPT 个 float，不是最理想的 coalesced pattern。v1.5 将数据分配方式改成 input4[j * TPB + tid]，让同一个 warp 在每次 load 时访问连续的 float4，从而得到更标准的 warp-level coalescing。

所以我选择 v1 这个版本作为 baseline，但明确猜想其 thread-local insert topk 不好。

## V2: 增大 VPT 的尝试

在 v1 的基础上，我尝试过一个方向：让每个 thread 处理更多元素。比如把 VPT 增大到 128，让每个 thread 扫描 128 个 float，然后维护一个长度为 32 的 local topk。

这个想法有一定道理。因为每个 block 处理的数据更多，block 数会减少，stage2 需要处理的 partial topk 数量也会减少，此外，每一个thread 处理更多元素，这个过程中会有更多元素会被丢弃，实现一些 warp select 的思想。

该版本参数是：

```
TPB = 256
VPT = 128
K   = 32
```

每个 block 处理 `256 * 128 = 32768 elements`, 相比 VPT=32 时每个 block 处理 8192 个元素，block 数减少了 4 倍。

但是实际性能似乎不好，大约只有 60GB/s 左右。原因是这个版本把太多工作塞进了单个 thread。每个 thread 要对 128 个元素维护 top32，每个元素都可能触发 insert merge。这样 thread 内串行指令量非常大，warp 内控制流也不够规则。

这个版本说明一个重要问题：减少 block 数并不一定能提升性能。如果代价是让每个 thread 做更多串行 selection，那么整体吞吐可能反而下降。GPU 更适合大量线程做比较规则的小任务，而不是让单个 thread 做很长的串行维护。

因此 VPT=128 + insert topk 这个方向被放弃。

## V1.5: VPT=32 + Thread-local Bitonic Sort

后面我换了一个思路。既然目标是 K=32，那可以让每个 thread 正好读取 32 个元素，也就是：

```
VPT = K = 32
```

这样每个 thread 读取的 32 个元素本身就是该 thread 的 local top32。我们不需要做 insert selection，只需要把这 32 个元素排序成降序。

这似乎有些冗余，因为 bitonic sort 是完整排序，而 topk 只需要 selection。但在这个固定场景下，bitonic sort 有一个优势：结构规则，循环边界固定，适合 compiler unroll。相比 insert topk 的 data-dependent branch 和移动，固定 compare-swap network 更适合 GPU。

v1.5 的参数是：

```
TPB = 256
VPT = 32
K   = 32
```

每个 block 处理 `TPB * VPT = 256 * 32 = 8192 elements`.

stage1 的流程现在如下：
```
每个 thread 读取 32 个 float
  ↓
thread-local bitonic sort
  ↓
得到该 thread 的 top32
  ↓
block 内 merge reduction
  ↓
输出 block top32
```
### Coalesced float4 Load

在这个版本里，一个重要优化是把 global load 改成 coalesced float4 load。

原来如果每个 thread 读取：
```
input[tid * VPT + i]
```
那么 warp 内 lane 之间的地址不是连续的。在这个版本中，我改为了：
```cpp
const float4* input4 = reinterpret_cast<const float4*>(input + block_start);

#pragma unroll
for (int j = 0; j < VPT / 4; ++j) {
    float4 v = input4[j * TPB + tid];

    local[4 * j + 0] = v.x;
    local[4 * j + 1] = v.y;
    local[4 * j + 2] = v.z;
    local[4 * j + 3] = v.w;
}
```
对于同一个 j，warp 内相邻 lane 访问的是连续的 float4，global memory load 应该 coalesced。

> warp level coalescing transaction 的粒度是 32 bytes, 因此我们在保证了 thread ldg128的同时，也保证了 warp 内的 coalesced load。

这个改动非常合理，因为 stage1 需要扫描完整 input。即使 topk 的瓶颈最后不是纯 DRAM bandwidth，coalesced load 仍然是基础。

### Shared Memory Padding

block 内 reduction 时，每个 thread 会把自己的 top32 写入 shared memory。自然的 layout 是：
```
topk[tid * K + i]
```
但是 K=32 时，这种 layout 很容易触发 shared memory bank conflict。因此我加了 padding：
```
LD = K + 1 = 33
topk[tid * LD + i]
```
这样每个 thread 的 top32 之间错开一个 float，避免很多规则性的 bank 冲突。

v1.5 初版性能已经明显提升。最大输入下大约是：
```
N = 268,435,456
time ≈ 3.260 ms
effective input bandwidth ≈ 306.7 GB/s
```
相比最早的 50GB/s，这已经是一个明显进步。但这个版本后面还暴露出两个问题：merge2TopK 还不够高效，stage2 依然是单 block 串行归约。

## V1.5.1: merge2TopK 从 while 改成 for

v1.5 之后，一个非常关键的小改动是我重写了 merge2TopK。

这个函数在 block 内 reduction 和 stage2 中会被大量调用。它的任务是把两个已经降序排列的 top32 合并成一个新的 top32。

原来的实现为 while 风格，逻辑正确，但对 compiler 不够友好。后来改成固定长度的 for loop，并加上 #pragma unroll：
```cpp
template <int K>
__device__ __forceinline__ void merge2TopK_merge(float* left, const float* right) {
    float result[K];

    int l = 0;
    int r = 0;

#pragma unroll
    for (int i = 0; i < K; ++i) {
        float lv = left[l];
        float rv = right[r];

        if (lv >= rv) {
            result[i] = lv;
            ++l;
        } else {
            result[i] = rv;
            ++r;
        }
    }

#pragma unroll
    for (int i = 0; i < K; ++i) {
        left[i] = result[i];
    }
}
```
这个修改带来的提升非常之大：
```
before:
  time = 3.260 ms
  bandwidth = 306.7 GB/s

after:
  time = 2.028 ms
  bandwidth = 493.2 GB/s
```
提升大约`3.260 / 2.028 ≈ 1.61x`!

这个结果说明，对于 K=32 这种固定小 K 算子，模板参数、固定 loop trip count、#pragma unroll、__forceinline__ 都非常重要。让 compiler 清楚地看到循环只跑 32 次，可以减少不少控制流和 loop overhead。

## V1.5 的一些负优化尝试

在 v1.5 主线附近，我还试过几个参数修改，但结果都不怎么好。

### VPT=16

第一个尝试是把 VPT 从 32 改成 16。直觉上，thread-local sort 会更小一些，因为每个 thread 只处理 16 个元素。

但实际结果明显变慢：
```
N = 268,435,456
time ≈ 4.355 ms
bandwidth ≈ 229.6 GB/s
```
原因是每个 block 处理的数据变少，block 数增加，block-level reduction 和 stage2 的相对开销变大。同时 VPT=16 和 K=32 不再自然匹配，局部 topk 结构也没有 VPT=K=32 那么干净。因此 VPT=16 被放弃。

### TPB=128

另一个尝试是把 block size 从 256 改成 128：
```
TPB = 128
VPT = 32
```
结果也更慢：
```
N = 268,435,456
time ≈ 5.052 ms
bandwidth ≈ 197.9 GB/s
```
这说明在当前 case 下，TPB=256 是比较合适的。它提供了足够的 block 内并行度，同时每个 block 处理 8192 个元素，stage1 和 stage2 的规模比较平衡。

### Two-level block reduction

我还尝试过把 block 内 reduction 改成两层结构：先在 warp 内 reduce，再跨 warp reduce。这个想法是减少 __syncthreads()，因为 warp 内可以用 __syncwarp()。

但是实际结果也更慢：
```
N = 268,435,456
time ≈ 3.697 ms
bandwidth ≈ 270 GB/s
```
原因是，block 内 reduction 的主要开销不是 barrier 数量，而是 merge 本身。两层 reduction 并没有减少 merge 次数，反而引入了额外的 warp-level 同步和更复杂的调度开销。最后还是回退到简单的 block-wide reduction。

这一阶段的结论是：当前最好的 stage1 结构仍然是：
```
TPB=256
VPT=32
K=32
float4 coalesced load
thread-local bitonic sort
AoS padded shared memory
for-unrolled merge2TopK
Radix Select Experiment
```

除了 merge / bitonic 这条路线，根据知乎文章的指引，我还尝试过 **radix select算法**。

radix select 的思路是从数值 bit pattern 的高位到低位逐步确定第 K 大值。每一轮统计不同 bucket 里的元素数量，然后判断第 K 大值落在哪个 bucket 中。大致流程是：
```
从最高位开始
  ↓
统计每个 radix bucket 的元素个数
  ↓
判断 kth largest 落在哪个 bucket
  ↓
更新 mask / desired pattern
  ↓
继续处理下一组 bit
```
PyTorch 的 topk 里也有类似 radixSelect 的思想。直接对全局 input 做 radix select 会反复扫描 global memory，开销很大。因此我实现的是 block-local radix select：
```
stage1:
  每个 block 读入 8192 个元素
  在 shared memory 里用 radix select 找本 block 的第 32 大 threshold
  收集本 block top32

stage2:
  合并所有 block top32
```

这个版本最后是正确的，但是性能不理想：
```
N = 268,435,456
time ≈ 10.310 ms
bandwidth ≈ 97.0 GB/s
```
它比一些 insert 版本好，但远远慢于 v1.5 / v1.6 的 bitonic + merge 路线。

原因主要是 radix select 每一轮都需要 bucket counting、warp ballot、popc、shared memory 汇总和同步。对于 K=32, VPT=32 这个固定 case，这些额外控制逻辑太多了。相比之下，每个 thread 直接 bitonic sort 32 个数，再做 merge reduction，虽然看起来“排序更多”，但结构更规则，实际更快。

这个版本最后没有继续优化。它的价值主要是确认：radix select 是一个更通用的 selection 方法，但在当前这个 highly specialized case 下不是最优路线。

## Profiling V1.5: 真正瓶颈在 Stage2

v1.5 for-merge 版本已经到了：
```
N = 268,435,456
time ≈ 2.028 ms
bandwidth ≈ 493.2 GB/s
```
这时如果只看整体时间，很容易继续对 stage1 做优化，比如继续改 local sort 或 shared memory layout。但 Nsight Systems 告诉我们一个很重要的信息：stage2 占了75%的时间。

v1.5 的 stage2 是一个单 block kernel：
```cpp
device_topk_fp32_stage2<<<1, TPB>>>(partial, output, blocks);
```
对于最大输入：
```
N = 268,435,456
TPB = 256
VPT = 32
elements_per_block = 8192
blocks = 32768
```
stage1 输出：
```
partial[32768, 32]
```
而 stage2 只有一个 block，需要在一个 block 内串行处理所有 partial topk。逻辑大概是：
```cpp
for (loop = 0; loop < BLK_NUM; loop += TPB) {
    load 256 rows partial topk
    reduce within one block
    merge into final result
}
```
所以这个单 block stage2 实际上要串行做 128 轮 block reduction。这个结构明显很不合理。

>为什么在一开始我没有优化？
>因为在 v1.5之前， 我没有增加max test case的N，stage2的问题并不突出，当时想着launch kernel的开销说不定比stage2的串行归约还大，所以没有去优化它。

Nsight Systems 显示，在这个版本里 stage2 占比非常高。虽然 profiler 下的绝对时间和普通 benchmark 不完全一致，但比例已经足够说明问题，stage2 的单 block 串行归约已经成为主要瓶颈。

这个 profiling 结果决定了下一版优化方向，也就是先把 stage2 并行化。

## V1.6: Multi-block Stage2 Reduction

v1.6 的核心改动就是把 stage2 从单 block 串行归约改成 multi-block hierarchical reduction。

原来的 stage2 是：
```
partial[32768, 32]
  ↓
1 block
  ↓
result[32]
```
新的 stage2 分成两步：
```
partial[32768, 32]
  ↓ stage2a: 128 blocks
partial2[128, 32]
  ↓ stage2b: 1 block
result[32]
```
stage2a 中，每个 block 负责 reduce 256 行 partial topk：
```
256 rows × 32 values
  ↓
1 row × 32 values
```
因此 32768 行 partial 会被 reduce 成：
```
32768 / 256 = 128 rows
```
然后 stage2b 再用一个 block reduce 这 128 行，得到最终 top32。

通用 reduce kernel 大致是：
```cpp
template<int TPB, int K>
__global__ void device_topk_fp32_stage2_reduce(
    const float* input,   // [rows, K]
    float* output,        // [ceil(rows / TPB), K]
    int rows
) {
    __shared__ float local[TPB * (K + 1)];

    int tid = threadIdx.x;
    int row = blockIdx.x * TPB + tid;

    for (int i = 0; i < K; ++i) {
        local[tid * LD + i] = row < rows ? input[row * K + i] : -inf;
    }

    __syncthreads();

    for (int stride = TPB / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            merge2TopK(local + tid * LD, local + (tid + stride) * LD);
        }
        __syncthreads();
    }

    if (tid == 0) {
        write output[blockIdx.x]
    }
}
```
host 侧变成：
```cpp
stage1<<<blocks, TPB>>>(input, partial, numel);

stage2_blocks = ceil(blocks / TPB);

stage2_reduce<<<stage2_blocks, TPB>>>(partial, partial2, blocks);
stage2_reduce<<<1, TPB>>>(partial2, result, stage2_blocks);
```
这个版本带来了整个优化过程里最大的一次提升：
```
v1.5 for-merge:
  time ≈ 2.028 ms
  bandwidth ≈ 493.2 GB/s

v1.6 multi-block stage2:
  time ≈ 0.526 ms
  bandwidth ≈ 2.0 TB/s
```
提升大约`2.028 / 0.526 ≈ 3.86x`.

> 这个提升也是很合理的，因为算一下在原先的nsys结果里 stage2占了 75% 的时间，stage1占了25%。新版的stage 2只占了 4% 左右，因此算一下加速比也大约是 1 / (0.25 + 0.05) = 3.85x，和实际 benchmark 非常接近。

这说明 stage2 现在很合理了。

## V1.6 Profiling: 瓶颈重新回到 Stage1

v1.6 之后，我重新用 Nsight Systems 查看 kernel 时间拆分。结果变成：
```
stage1:              522.692 us, 95.8%
stage2_reduce x 2:    22.752 us,  4.2%
```
这个结果说明 stage2 已经基本不是问题，瓶颈重新回到了 stage1。

再用 Nsight Compute 看硬件指标：
```
stage1:
  dram throughput = 15.31%
  sm throughput   = 71.88%
  registers/thread = 48
  local load = 0
  local store = 0
  active warps = 60.58%
```
这里有几个重要现象。

- local load = 0 和 local store = 0 说明没有 register spill。也就是说，local[32] 和 result[32] 都被 compiler 放在寄存器里，没有溢出到 local memory。

- dram throughput 只有 15% 左右，而 sm throughput 接近 72%。这说明当前 kernel 不是 DRAM bandwidth-bound，而是 SM/instruction-bound 或 shared-memory-bound。

- stage2_reduce 的耗时已经很小，即使继续优化 stage2，也不会带来很大收益。后续如果继续优化，重点应该放在 stage1.

因此找到了以下几种方向：
1. thread-local bitonic sort 的指令数
2. block-level merge 的 shared memory 访问
3. shared memory bank conflict
4. __syncthreads 开销

## V1.7 Attempt: SoA Shared Memory Layout

在 v1.6 的 ncu 结果里，stage1 仍然有比较多 shared memory bank conflict：

```text
shared load bank conflicts = 20,666,167
shared store bank conflicts = 867,656
```

因此我尝试继续修改 shared memory layout。v1.6 使用的是 AoS padded layout：

```cpp
LD = K + 1 = 33
topk[tid * LD + i]
```

也就是每个 thread 的 top32 基本连续存放，只是在每个 thread 后面加一个 padding。这样做的目的是避免 `K=32` 时过于规则的 bank mapping。

一个自然的想法是把 layout 改成 SoA：

```cpp
topk[i * TPB + tid]
```

也就是同一个 rank 的值放在一起。理论上，如果一个 warp 内所有 lane 同时访问同一个 rank，那么这种 layout 会让 bank 分布更加均匀。因此，从降低 shared memory bank conflict 的角度看，SoA 是一个值得尝试的方向。

不过这个版本最后没有得到可靠结论。初次测试时，SoA 版本的性能明显差于 v1.6 的 AoS padded 版本，但后来发现当时 GPU 并不是空载状态，因此这组数据不能作为有效 benchmark。由于短时间内等不到稳定的空闲 GPU，我没有继续验证这个版本，所以最终没有把 SoA 作为正式结论，也没有把它写成负优化。

从理论上看，SoA 是否有效本来也不能只靠 layout 静态判断。`merge2TopK` 的 shared memory 访问是 data-dependent 的：

```cpp
lv = left[l];
rv = right[r];
```

其中 `l` 和 `r` 会随着数据变化而变化。不同 lane 的 `l/r` 可能并不一致，因此 SoA 在规则访问下可能更均匀，但在实际 merge 过程中未必一定更好。它可能减少某些 bank conflict，也可能因为访问模式更分散而引入其他 shared memory replay 或指令开销。

因此，SoA 这一版目前只能算一个 tentative attempt。它提出了一个合理假设：通过改变 shared memory layout 来减少 bank conflict。但由于 benchmark 环境不稳定，这个假设还没有被有效验证。当前最终路线仍然保留 v1.6 的 AoS padded layout：

```cpp
topk[tid * 33 + i]
```

后续如果继续优化 stage1，可以重新在空闲 GPU 上测试 SoA 或其他 shared memory layout，并同时观察 kernel time、bank conflict、shared replay 和 SM throughput，而不是只看单个 metric。

---

## Performance Summary

整个版本演进大致如下：

```text
V1:
  merge insert
  ~50 GB/s

V2:
  VPT=128 + insert local topk
  ~60 GB/s
  线程内串行工作太多，放弃

V1.5:
  VPT=32, K=32
  thread-local bitonic sort
  coalesced float4 load
  shared padding
  ~306 GB/s

V1.5.1:
  merge2TopK while -> for unroll
  ~493 GB/s

V1.5 negative attempts:
  VPT=16
  TPB=128
  two-level block reduction
  均为负优化

Radix v3:
  block-local radix select
  ~97 GB/s
  正确但控制逻辑太多

V1.6:
  multi-block hierarchical stage2
  ~2.0 TB/s

V1.7 tentative:
  shared layout AoS padded -> SoA
  测试时 GPU 非空载，数据不可靠，暂不下结论
```

最大输入 `N = 268,435,456` 下，各版本大致结果：

| Version        | Main idea                    |         Time | Effective BW |
| -------------- | ---------------------------- | -----------: | -----------: |
| V1             | merge insert                 |       20ms 级 |      ~50GB/s |
| V2             | VPT=128 insert topk          |       15ms 级 |      ~60GB/s |
| Radix v3       | block-local radix select     |    10.310 ms |     97.0GB/s |
| V1.5 old merge | bitonic + while merge        |     3.260 ms |    306.7GB/s |
| V1.5 for merge | bitonic + for-unrolled merge |     2.028 ms |    493.2GB/s |
| V1.6           | multi-block stage2           |     0.526 ms |     ~2.0TB/s |
| V1.7 tentative | SoA shared layout            | not reliable | inconclusive |
| torch.topk     | generic topk                 |      ~6.4 ms |     ~155GB/s |
| torch.sum      | read bandwidth baseline      |     0.181 ms |     ~5.9TB/s |
| out.copy_      | copy bandwidth baseline      |     0.330 ms |     ~6.5TB/s |

---

## What Each Iteration Taught

这次优化过程中，每一版其实都对应一个比较明确的结论。

V1 说明，最自然的 insert topk 写法不适合这个 GPU 场景。它有太多 thread 内串行逻辑和 data-dependent control flow。

V2 说明，单纯增大 VPT、减少 block 数不是万能的。如果每个 thread 的局部 selection 成本太高，性能会更差。

V1.5 说明，在 `K=32` 这种固定小 K 场景下，完整 local bitonic sort 不一定比 selection 慢。规则的 fixed-size compare-swap network 反而更适合 GPU。

`merge2TopK` 的改写说明，小函数的写法会显著影响性能。对于固定 K，应该尽量模板化、unroll，并避免不必要的动态循环结构。用for loop 替代 while，能让 compiler 更好地优化。

VPT=16、TPB=128 和 two-level reduction 的失败说明，参数调优不能只凭直觉。减少单个 thread 的工作、减少 block size、减少 barrier，都不一定带来提升。必须看实际瓶颈在哪里。

Radix v3 说明，更通用的算法不一定适合特化场景。radix select 在通用 topk 中有价值，但在这个 `fp32, 1D, K=32, values-only` 场景下控制逻辑太重。

V1.6 指明了跨 block reduction 不能随便用单 block 串行处理。一旦 partial 数量很大，就应该做 hierarchical reduction。这是整个过程中最重要的结构性优化。而且要关注 nsys / ncu 的 profile，确认瓶颈在哪里，而不是只看 benchmark 的整体时间。

V1.7 的 SoA 尝试没有得到可靠结论。它提出了一个合理方向：通过改变 shared memory layout 来降低 bank conflict。但由于测试时 GPU 并非空载，结果不能用于判断这个方向是否有效。sub-millisecond kernel 对 benchmark 环境非常敏感；当 GPU 上存在其他 workload 时，单次结果可能完全失真。因此 SoA 目前只能算待验证方向，而不是负优化结论。

---

## Final Kernel Structure

最终保留下来的 v1.6 结构是：

```text
stage1:
  grid = numel / (TPB * VPT)
  block = 256 threads

  每个 thread:
    coalesced float4 load 32 个 fp32
    local bitonic sort
    写入 shared topk[tid * 33 + i]

  block 内:
    merge reduction
    输出 block top32

stage2a:
  多个 block 并行 reduce partial topk
  partial[32768, 32] -> partial2[128, 32]

stage2b:
  单 block reduce partial2
  partial2[128, 32] -> result[32]
```

关键参数：

```text
TPB = 256
VPT = 32
K   = 32
LD  = 33
```

关键优化点：

```text
1. VPT=K=32，让 local topk 变成 fixed-size local sort
2. float4 coalesced load
3. AoS padded shared memory layout
4. merge2TopK fixed for-loop + unroll
5. stage2 multi-block hierarchical reduction
```

---

## Conclusion

这次 topk kernel 优化从最早大约 50GB/s 的版本开始，最终优化到了大约 2TB/s effective input bandwidth。整个过程中，真正有效的优化并不是某个单独技巧，而是逐步定位瓶颈之后做出的结构性修改。

最重要的几次提升分别是：

```text
1. 从 insert topk 转向 VPT=32 local bitonic sort
2. global load 改成更规整的 float4 load pattern
3. merge2TopK 从 while 改成 fixed for-loop
4. stage2 从 single-block reduction 改成 multi-block hierarchical reduction
```

尤其是 v1.6 的 stage2 改造，是整个过程中最关键的一步。它把原本单 block 串行处理 32768 个 partial topk 的逻辑，改成了并行分层归约，直接把性能从 2.028ms 提升到 0.526ms。

后续如果继续优化，主要方向应该集中在 stage1。因为 v1.6 profile 已经显示：

```text
stage1: 95.8%
stage2: 4.2%
```

也就是说，stage2 已经不是主要问题。下一步可能要研究的是如何减少 thread-local bitonic sort 的指令量，或者尝试更细粒度的 warp-level selection / block-level selection 结构。

这次优化最大的收获是，CUDA kernel 优化不能只靠直觉。每一次改动都需要回答三个问题：

```text
为什么这个地方可能是瓶颈？
怎么改能减少这个瓶颈？
benchmark 和 profiler 是否支持这个判断？
```

有些优化看起来合理，比如 SoA shared layout，但如果 benchmark 环境不稳定，就不能贸然下结论；有些改动看起来只是小改写，比如 `merge2TopK` 从 while 改成 for，却能带来 1.6x 提升。最终能跑到 2TB/s，靠的不是一次大改，而是每一步都尽量让假设、实现、benchmark 和 profile 对齐。


## The end
本文写于 2026-7-7 的早上7点零七分（至少写这句话的时候是），主要用GPT 5.5辅助了写作（所以有很多AI味请见谅）修正也修不完全。让我们抖擞精神，开始写今天的gemm kernel优化吧。

## References

- https://zhuanlan.zhihu.com/p/1887974210787312715
- https://leimao.github.io/blog/CPU-TopK-Algorithm/
- Li, Y. et al. RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection. in Proceedings of the 38th ACM International Conference on Supercomputing 537–548 (2024). doi:10.1145/3650200.3656596.
