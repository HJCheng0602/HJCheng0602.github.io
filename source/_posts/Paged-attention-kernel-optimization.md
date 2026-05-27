---
title: Paged attention kernel optimization(I)
date: 2026-05-22 18:34:24
tags: 
    - CUDA
    - kernel optimization
categories:
    - blog
---
## Introduction
During recent interviews, I have been asked a question about my experience with optimizing 
complex kernels or experience on architecture higher than sm89. Unfortunately, I don't have much experience on this topic, but a few days ago, I have earned a H100x8 server for some reason, so I deside to spend a weekend to optimize my decode paged attention kernel in [NanoPD](https://www.hjcheng0602.cn/blog/nanopd/). The post is the first part of the optimization process, which contains my reading notes on the vLLM and flash infer kernels.

## vLLM's paged attention kernel
First let us admire the vLLM's paged attention kernel, which is implemented in [this file](https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cuh). In the following part, I will line by line analyze the code and try to understand the optimization techniques used in this kernel.

### block sum kernel
Ahead of directly analysing the paged attention kernel, we need to understand the block sum kernel:
```Cpp
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }
  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }
  // Make sure the data is in shared memory.
  __syncthreads();
  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }
  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }
  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}
```
First reduce in a warp, then store the result in shm, then reduce in the warp again, which is a common reduction pattern. No bank conflict as Warp Size is 32 or less, and the shared memory is float array, so each element is 4 bytes, which means each warp will write to a different bank. The reduction in the warp is done by shuffle instructions, which can be very efficient.But warp divergence can happen in the reduction, but it is not a big problem because the reduction is done in log2(NUM_WARPS) steps, which is small. The final result is broadcasted to all threads in the block, which can be efficient if NUM_WARPS is small.

### paged attention kernel
Now the main dish comes, first let us look at the kernel signature:
```Cpp
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step)
```
First consider the template parameters:
```cpp
typename scalar_t; // Q/K/V data type, can be float16, bfloat16, int8, etc.
typename cache_t; // KV cache store data type, can be float16, bfloat16, used for quant
int HEAD_SIZE; // head size, can be 64, 128, 256, etc.
int BLOCK_SIZE; // the block size of paged attention, can be 128, 256, etc.
int NUM_THREADS; // the number of threads per block, can be 128, 256
vllm::Fp8KVCacheDataType KV_DTYPE; // kv quantization data type, can be int8, int4, etc.
bool IS_BLOCK_SPARSE; // whether the attention is block sparse, if true, the block_tables will be used to determine which blocks are valid.
int PARTITION_SIZE; // the partition size of the attention.
```
> **Partition attention**:
> At the decode stage, the query may be fewer than the prefill stage, but we still need to tackle a sequence-long kv cache. So the idea is to cut the kv cache into multiple partitions, and each partition will be processed by one kernel. Thus we achieved better parallelism and memory access pattern. The partition size can be tuned for better performance, and it is usually set to be a multiple of the block size.(from Tri Daos' 2023 work [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html))

Then consider the grid and block configuration:
The comment said `Grid:(num_heads, num_seqs, max_num_partitions)`:
```
blockIdx.x: head index, range from 0 to num_heads-1
blockIdx.y: sequence index, range from 0 to num_seqs-1
blockIdx.z: partition index, range from 0 to max_num_partitions-1
```
Each cuda block will process one sequence of one head for one partition.

#### Output parameters:
```Cpp
float * __restrict__ exp_sums; // [num_seqs, num_heads, max_num_partitions]
float * __restrict__ max_logits; // [num_seqs, num_heads, max_num_partitions]
scalar_t * __restrict__ out; // [num_seqs, num_heads, max_num_partitions, head_size]
```
They are the middle results of Flash-Decoding. When `PARTITION_SIZE > 0`, each partition computes local softmax, expsum, weighted ouput and then reduce them to get the final output. `PARTITION_SIZE = 0` means no partition, the kernel will compute the final output directly without reduction.

#### Input parameters:
```Cpp
scalar_t * __restrict__ q; // [num_seqs, num_heads, head_size]
cache_t * __restrict__ k_cache; // [num_blocks, num_kv_heads, head_size/x, block_size, x]
cache_t * __restrict__ v_cache; // [num_blocks, num_kv_heads, head_size, block_size]
int num_kv_heads; 
float scale;
int * __restrict__ block_tables; // [num_seqs, max_num_blocks_per_seq]
int * __restrict__ seq_lens; // [num_seqs]
int max_num_blocks_per_seq;
float * __restrict__ alibi_slopes; // [num_heads]
int q_stride, kv_block_stride, kv_head_stride;
float *k_scale, *v_scale;
int tp_rank;
int blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size, blocksparse_head_sliding_step;
```
- `q`: the query tensor, each sequence has one query vector for each head, so the shape is [num_seqs, num_heads, head_size].
- `k_cache` and `v_cache`: the kv cache tensor, each sequence has multiple blocks of kv cache, each block has multiple kv heads, each kv head has multiple key/value vectors, so the shape is [num_blocks, num_kv_heads, head_size/x, block_size, x], where we split head_size into x parts to vectorize the memory access for better performance. The actual head size is `head_size/x * x = head_size`, and the actual block size is `block_size * x`.
- `num_kv_heads`: the number of kv heads for each head.
- `scale`: the scaling factor for the attention, usually set to be `1/sqrt(head_size)`.
- `block_tables`: the block table for block sparse attention, each sequence has a block table to indicate which blocks are valid, so the shape is [num_seqs, max_num_blocks_per_seq].
- `seq_lens`: the actual sequence length for each sequence, so the shape is [num_seqs].
- `max_num_blocks_per_seq`: the maximum number of blocks for each sequence, used for block sparse attention.
- `alibi_slopes`: the alibi slopes for each head when we don't use RoPE, we do not discuss it here, so just ignore it.
- `q_stride`, `kv_block_stride`, `kv_head_stride`: the stride for accessing the q, k_cache and v_cache tensors, avoid calculating the stride in the kernel for better performance.
- `k_scale` and `v_scale`: the scaling factor for the quantized k and v, used for unquantization.
- `tp_rank`: the tensor parallel rank of the current process, used for partitioning the kv cache for tensor parallelism.
- `blocksparse_local_blocks`, `blocksparse_vert_stride`, `blocksparse_block_size`, `blocksparse_head_sliding_step`: the parameters for block sparse attention, used for calculating the valid blocks for each head.

Next comes the kernel body, first to decide the work space of now block:
```Cpp
const int seq_idx = blockIdx.y;
const int partition_idx = blockIdx.z;
const int max_num_partitions = gridDim.z;
constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
const int seq_len = seq_lens[seq_idx];
```
Read the sequence index and partition index from the block index, and read the sequence length from the `seq_lens` tensor.

Then calculate the start and end position of the current partition:
```cpp
const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
```
Calculate how many blocks are there for the current sequence, which is the sequence length divided by the block size, rounded up.

```cpp
const int num_blocks_per_partition =
    USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;
```
Calculate how many blocks are there for each partition.
```cpp
const int start_block_idx =
    USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
const int end_block_idx =
    MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
const int num_blocks = end_block_idx - start_block_idx;
```
Obviously, we can infer from the code itself.

Then convert the block range to the token range:
```cpp
const int start_token_idx = start_block_idx * BLOCK_SIZE;
const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
const int num_tokens = end_token_idx - start_token_idx;
```

#### Thread Group Design
```cpp
constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
```
A thread group is a group of threads that work together to process the QK product for one token.

One warp has 32 threads and a KV block has `BLOCK_SIZE` tokens, so the design is to let threads balance the workload of one block. Each token receives `WARP_SIZE / BLOCK_SIZE` threads to compute the QK product.

```cpp
constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
```
Calculate how many thread groups are there in one block, which is the total number token can be processed in parallel.

```cpp
constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
```
If `BLOCK_SIZE > WARP_SIZE`, each thread group will process multiple tokens.

```cpp
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
const int thread_idx = threadIdx.x;
const int warp_idx = thread_idx / WARP_SIZE;
const int lane = thread_idx % WARP_SIZE;
```
Standard thread index calculation.

#### GQA head projection
```cpp
const int head_idx = blockIdx.x;
const int num_heads = gridDim.x;
const int num_queries_per_kv = num_heads / num_kv_heads;
const int kv_head_idx = head_idx / num_queries_per_kv;
```
GQA means multi Q head with shared K/V, `num_queries_per_kv` is the number of query heads that share the same kv head, so we can calculate the kv head index from the query head index.

#### Vector Type Definition
```cpp
constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
```
The goal is to vectorize the memory access in a thread group to read 16 bytes of data which targeting `LDG.128` instruction.

For an example: `THREAD_GROUP_SIZE = 4`, `scalar_t = float16`, then `VEC_SIZE = 16 / (4 * 2) = 2`, which means each thread will read 2 float16 elements once, which is 4 bytes, and the whole thread group will read 16 bytes once.

Then we can define the vector type for the q/k/v:
```cpp
using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;
```

Then we can calculate how many elements in one thread:
```cpp
constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;
```
`HEAD_SIZE` elements is assigned to one thread group, each thread process `NUM_ELEMS_PER_THREAD` elements. Then make groups according to the `VEC_SIZE` for vectorized memory access, each thread will process `NUM_VECS_PER_THREAD` vectors.

#### Coordinate in the Thread Group
```cpp
const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;
```
Calculate the thread group index and the offset of the thread in the thread group, which will be used for memory access and reduction.  

#### Load the Query to registers.
```cpp
  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs
```
`Q`'s shape is `[num_seqs, num_heads, head_size]` and the grid shape is `[num_heads, num_seqs, max_num_partitions]`, so the query for the current block can be calculated by `q +seq_idx * q_stride + head_idx * HEAD_SIZE`. The shared memory `q_vecs` layout is suitable for the thread group design.

Then each thread in the thread group(assume it's size is 4) will traverse a row of the shm, thread 0, 4, 8, 12 will traverse the first row, then their thread group idx is 0, 1, ... NUM_THREAD_GROUPS - 1, we assign them to a row of the shm and repeat it. For example, if `NUM_THREAD_GROUPS = 4`, then thread 0, 4, 8, 12 will traverse(0, 1, 2, 3) , (4, 5, 6, 7)...
The Q access model is computed by `vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE`, meaning that the first row of shm stores (0, 4, 8, 12), the second row stores (1, 5, 9, 13), and so on. This access pattern can ensure coalesced memory access for the query.

> Why we store the Query in shared memory instead of registers?
> The query will be reused for multiple thread groups. After storing it into shm, each thread can access the query of other thread groups for free.


#### Shared Memory Plan
```cpp
extern __shared__ char shared_mem[];
float* logits = reinterpret_cast<float*>(shared_mem);
__shared__ float red_smem[2 * NUM_WARPS];
```
The `shared_mem` is used to store the attention score of all the tokens and the `red_smem` is used for the reduction.

```cpp
constexpr int x = 16 / sizeof(cache_t);
```
K cache's layout is `[num_blocks, num_kv_heads, head_size/x, block_size, x]`, so we need to calculate the `x` for vectorized memory access.
```cpp
float qk_max = -FLT_MAX;
```
Each thread keep a local max used for online softmax.

```cpp
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
```
`block_tables`'s layout is `[num_seqs, max_num_blocks_per_seq]`, we get the block table information.

Here we would not consider the block sparse case, so just ignore the block sparse related code for now.

#### Main Loop
```cpp
for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t* k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else {
          // Vector conversion from Quant_vec to K_vec.
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
              k_vec_quant, *k_scale);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }
```

##### KV block assign loop
```cpp
for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
     block_idx += NUM_WARPS) 
```
Each warp tackle a few KV blocks, if there are 4 warps and 16 blocks, then the warp 1 will tackle block 1, 5, 9, 13.

##### Physical block location
```cpp
const int64_t physical_block_number =
    static_cast<int64_t>(block_table[block_idx]);
```
The block table is used to map the logical block index to the physical block index in the KV cache, which is used for block sparse attention. 

##### Inner loop assign token in a warp
```cpp
for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
    const int physical_block_offset =
        (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
```
Usually, `NUM_TOKENS_PER_THREAD_GROUP` is 1, which means each thread group process one token. 

##### Load K 
```cpp
K_vec k_vecs[NUM_VECS_PER_THREAD];

for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
    const cache_t* k_ptr =
        k_cache + physical_block_number * kv_block_stride +
        kv_head_idx * kv_head_stride + physical_block_offset * x;
    const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
    const int offset1 = (vec_idx * VEC_SIZE) / x;
    const int offset2 = (vec_idx * VEC_SIZE) % x;
```
K cache's layout is `[num_blocks, num_kv_heads, head_size/x, block_size, x]`, so the base ptr is located to `[physical_block_number, kv_head_idx, 0, physical_block_offset, 0]`. This is the start position of the k vector for the current token. Then the `vec_idx` is used like the Q loading part to ensure `thread_group_offset` thread process the `vec_idx`th vector.

```cpp
k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
```
Then we can load the k vector to registers. 
Assume `THREAD_GROUP_SIZE=4, VEC_SIZE=2, x=8, NUM_VECS_PER_THREAD=8`, a thread group will process a token as below:
```
thread_group_offset=0, j=0: vec_idx=0, offset1=0, offset2=0 →  [0,1]
thread_group_offset=1, j=0: vec_idx=1, offset1=0, offset2=2 →  [2,3]
thread_group_offset=2, j=0: vec_idx=2, offset1=0, offset2=4 →  [4,5]
thread_group_offset=3, j=0: vec_idx=3, offset1=0, offset2=6 →  [6,7]

thread_group_offset=0, j=1: vec_idx=4, offset1=1, offset2=0 →  [8,9]
thread_group_offset=1, j=1: vec_idx=5, offset1=1, offset2=2 →  [10,11]
...
```

#### Dot product and store the logits
```cpp
// Compute dot product.
// This includes a reduction across the threads in the same thread group.
float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                         q_vecs[thread_group_offset], k_vecs);
```
Each thread compute the dot product and reduce in the thread group by `Qk_dot`, which is implemented by warp shuffle. 

```cpp
if (thread_group_offset == 0) {
    const bool mask = token_idx >= seq_len;
    logits[token_idx - start_token_idx] = mask ? 0.f : qk;
    qk_max = mask ? qk_max : fmaxf(qk_max, qk);
}
```
Then store the logits to shared memory, and update the local max for softmax. 

#### Reduction for max logits
```cpp
// Perform reduction across the threads in the same warp to get the
// max qk value for each "warp" (not across the thread block yet).
// The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();
```

Reduce all the kv block processed by a warp.

```cpp
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);
```

Then cross warps to get the max qk value.

Then compute the exp and the exp sums:
```cpp
float exp_sum = 0.f;
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
}

exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);
```

After this, we normalize the logits and get the softmax output:
```cpp
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
}
__syncthreads();
```

#### Load V into the memory
V cache's layout is `[num_blocks, num_kv_heads, head_size, block_size]`, which is different from K cache. 

```cpp
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
```
There are `BLOCK_SIZE` elements in one row of V, split them in `V_VEC_SIZE`. A warp has `WARP_SIZE` threads, so one iteration can process `NUM_ROWS_PER_ITER` rows, and each thread will process `NUM_ROWS_PER_THREAD` rows.


#### acc initialization
```cpp
  // Initialize the accumulators.
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }
```

Then comes the second main part: the matrix multiplication between the softmax output and the V vectors.

```cpp
scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
          !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
        continue;
      }
    }
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                           start_token_idx));

    const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                           kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else {
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          // Vector conversion from V_quant_vec to V_vec.
          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec,
                                                                    *v_scale);
        }
        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }
```

This code is similar to the previous loop for K.
