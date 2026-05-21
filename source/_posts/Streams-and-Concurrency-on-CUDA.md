---
title: Streams and Concurrency on CUDA
date: 2026-05-21 15:27:54
description: "Streams and Concurrency on CUDA"
tags: 
  - CUDA
categories:
    - blog
---

## Introduction
I have learned CUDA kernel programming for a long time, but I have never learnt CUDA streams, only knowing that CUDA streams can be used to achieve concurrency. Today by reading the [NVIDIA slides](https://developer.download.nvidia.cn/CUDA/training/StreamsAndConcurrencyWebinar.pdf) on CUDA streams, I have a better understanding of CUDA streams and concurrency. 

## Default stream
By default, all CUDA operations are issued into a single stream, called the default stream. Operations in the default stream are executed sequentially, and they are not concurrent with any other operations.The special behavior of the default stream is that it is wholely sync for host and device, which means each time we submit a operation to the default stream, the host will insert an implicit `cudaDeviceSynchronize()` after and before the operation. But there are several exceptions to this rule:
- Kernel launches in the default stream are asynchronous with respect to the host, but they are still serialized with respect to other operations in the default stream.
- `cudaMemcpyAsync()` and `cudaMemsetAsync()` operations in the default stream are asynchronous with respect to the host.
- `cudaMemcpy()` in the same device.
- `cudaMemcpy()` below **64KB** between host and device.

## Requirements for Concurrency
To achieve concurrency, we need to meet the following requirements:
- Use non-default streams for concurrent operations.
- `cudaMemcpyAsync()` with host from pinned memory.
- sufficient resources must be available on the device to execute concurrent operations.

## Some examples
```Cpp

cudaMalloc(&dev1, size);
double * host1 = (double *)malloc(&host, size);
...
cudaMemcpy(dev1, host1, size, cudaMemcpyHostToDevice);
kernel2<<<grid, block, 0>>>(..., dev2, ...);
kernel3<<<grid, block, 0>>>(..., dev3, ...); 
cudaMemcpy(host4, dev4, size, cudaMemcpyDeviceToHost);
```
Above code will be executed synchronously, because all operations are issued into the default stream. Observing the nsys timeline, we can see that all operations are executed sequentially.

<div>
    <img src="sync.png" alt="nsys timeline 1" style="width: 100%;">
</div>

```Cpp
cudaStream_t streams[NUM_STREAMS];
for(int i = 0; i < NUM_STREAMS; i ++){
    cudaStreamCreate(&streams[i]);
}
for(int i = 0; i < NUM_STREAMS; i ++){
    int offset = i * chunk;
    cudaMemcpyAsync(dev1 + offset, host + offset, chunk_size, cudaMemcpyHostToDevice, streams[i]);
}
for(int i = 0; i < NUM_STREAMS; i ++){
    int offset = i * chunk;
    kernel1<<<(chunk + 255) / 256, 256, 0, streams[i]>>>(dev1 + offset, dev2 + offset, chunk);
}
for(int i = 0; i < NUM_STREAMS; i ++){
    int offset = i * chunk;
    cudaMemcpyAsync(host + offset, dev2 + offset, chunk_size,
                    cudaMemcpyDeviceToHost, streams[i]);
}
```
Above code will be executed concurrently, because we have issued operations into different streams. Observing the nsys timeline, we can see that all operations are executed concurrently.
<div>
    <img src="async.png" alt="nsys timeline 2" style="width: 100%;">
</div>

Another overlap example is as follows:
```Cpp
cudaMemcpy(dev1, host1, size, H2D);      
kernel2<<<grid, block>>>(dev2);           // launch kernel is asynchronous with respect to the host.
some_CPU_method();                        // overlap with kernel2
kernel3<<<grid, block>>>(dev3);           
cudaMemcpy(host4, dev4, size, D2H);      
```
In above code, `kernel2` will be launched asynchronously with respect to the host, so `some_CPU_method()` can be executed concurrently with `kernel2`. However, `kernel3` and `cudaMemcpy()` will be executed sequentially after `kernel2`, because they are issued into the default stream. 

## Explicit Synchronization
- Synchronize everything: `cudaDeviceSynchronize()` : blocks host until all issued CUDA operations are completed.
- Synchronize a stream: `cudaStreamSynchronize(stream)` : blocks host until all operations in the specified stream are completed.
- Synchronize using Events: `cudaEventSynchronize(event)` : blocks host until the specified event is completed. Events can be used to measure the time between operations in different streams.

### Some Event Using Examples
```Cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaMemcpyAsync(dev1, host1, size, H2D, stream1);
cudaEventRecord(start, stream1); // record start event after memcpy

cudaMemcpyAsync(host2, dev2, size, D2H, stream2);
cudaStreamWaitEvent(stream2, start, 0); // make stream2 wait for the start event
kernel<<<grid, block, 0, stream2>>>(...); // kernel will execute after the start event is recorded
cudaEventRecord(stop, stream2); // record stop event after kernel launch
cudaEventSynchronize(stop); // wait for the stop event to complete
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop); // calculate elapsed time between start and stop events
printf("Elapsed time: %f ms\n", elapsedTime);
```

## Implicit Synchronization
Some operations will cause implicit synchronization, without knowing it, we may introduce unexpected synchronization points in our code, which can lead to performance degradation. Some examples of implicit synchronization are as follows:
- `cudaMallocHost()/cudaFreeHost()`: These functions will block the host until all previously issued CUDA operations are completed, because they need to ensure that the pinned memory is not being used by any ongoing CUDA operations.
- `cudaMalloc()`: This function will block the host until all previously issued CUDA operations are completed, because it needs to ensure that there are sufficient resources available on the device to allocate the requested memory.
- `cudaMemcpy()`: This function  needs to ensure that the data transfer is not being interfered by any ongoing CUDA operations.
- `cudaDeviceSetCacheConfig()`: This function needs to ensure that the cache configuration is not being changed while any ongoing CUDA operations are using the cache, so it will block the host until all previously issued CUDA operations are completed.

The right way to avoid implicit synchronization is to assign the memory allocation and deallocation in the beginning and the end of the program, and use `cudaMemcpyAsync()` instead of `cudaMemcpy()` for data transfer between host and device.

## Stream Scheduling
Take Fermi architecture as an example, it has 3 queues: 1 compute engine queue, 2 copy engine queues (one for H2D and one for D2H). 

The shedule rule is as follows:
CUDA operations are pushed into the target queue based on the type of operation in the launch order. One operation is issued only when the three conditions are met:
- In the same stream, all previously issued operations have been completed.
- Ahead of the operation in the same queue, there is no other operation that is still executing.
- The resources required for the operation are available on the device.

**One blocked operation can block the entire queue even there are other operations in the queue belonging to different streams.** So the launch order of operations can affect the performance of the program. 

An example of stream scheduling is as follows:

<div>
    <img src="streamscheduling1.png" alt="stream scheduling1" style="width: 100%;">
</div>

<div>
    <img src="streamscheduling2.png" alt="stream scheduling2" style="width: 100%;">
</div>


## Concurrent Kernel Scheduling
Normally, a signal is inserted into the queues, after the operation is issued, to indicate the completion of the operation. But for the compute engine queues, when compute kernels are issued sequentially, the signal is not inserted until the kernel is completed. So if there are multiple kernels issued into the compute engine queue, they will be executed sequentially, even if they belong to different streams.

In some situations this delay of signals can block other queues.

## Conclusion
Maybe the slides I read is a bit old, but it still gives me a good understanding of CUDA streams and concurrency. I will try to use CUDA streams in my future projects to improve the performance of my code.