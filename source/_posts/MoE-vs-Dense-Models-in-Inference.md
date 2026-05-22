---
title: MoE vs Dense Models in Inference
categories:
  - readings
tags:
  - null
bibtex: |
  @misc{epoch2024moevsdensemodelsinference,
  title={{How do mixture-of-experts models compare to dense models in inference?}},
  author={Ege Erdil},
  year={2024},
  url={https://epoch.ai/gradient-updates/moe-vs-dense-models-inference},
  note={Accessed: 2026-05-21}}
references:
  - ''
date: 2026-05-21 22:12:11
---

## Introduction
In a recent review conducted by Mimo, the reviewer asked me a question: "How do mixture-of-experts (MoE) models compare to dense models in inference and implementation?" I thought a while and answered according to my intuition, but obviously the answer was not satisfactory. So I decided to do some research and write this post to share my findings.

This post is based on the article [How do mixture-of-experts models compare to dense models in inference?](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference) by Ege Erdil, published on May 21, 2024. The article provides a detailed analysis of the differences between MoE and dense models in terms of inference and implementation.

The first main part of the article discusses the advantages of MoE models from an inference perspective:
- MoE models have fewer parameters than dense models, which can lead to faster inference times and lower memory usage.
- MoE models tend to be shallower and wider than dense models, which can also contribute to faster inference times.
- MoE models tend to have smaller attention blocks, i.e. the product of their number of attention heads with the head dimension is smaller, but whether this happens depends on whether we use GQA or MQA.

## MoE has fewer parameters than dense models
The most discussed advantage of MoE models is that they do less arithmetic than dense models, as each token will only be processed by a subset of the model's parameters. If the computation is the constraint of the system, then MoE models can be more efficient than dense models. 

However, as we diving into the details, compute bound is not the only constraint of the inference process. 

During the **Prefill** stage, we can compute all the token in parallel. In this case, the network latency and memory bandwidth can't be the main bottleneck, because the batch size of the tokens is large enough to allow us hide them behind the computation. So in this case, MoE models can be more efficient than dense models.

But in the **Decode** stage, if we still use large batch size and few GPU, the above situation still holds. However, if we want to going fast, we need to use more GPU to eplit arithmetic and memory workload known as TP. Extra network communication is required to synchronize the TP, which can be a bottleneck for MoE models. Briefly, as the number of GPU increases, the arithmetic workload per GPU decreases, and the network communication overhead becomes more significant. In the practical setting, the number of GPU is usually large enough to make the network communication overhead a bottleneck for MoE models.So in this case, MoE models can be no more efficient than dense models.

The article takes an example of a Llama 3.3 70B model with TP=8, H100 single node. In the ffn blocks alone, each token processed will require 8192 width vector to be all-reduced for each layer, as 8192 is the model dimension. Assume our quant precision is 16 bits, and the llama model has 80 layers, then the total communication of each token will be 8192 * 80 * 16 bits = 1.25 MB. At the critical batch size of 300 tokens, the total communication will be 384 MB, while the NVLink all-reduce bandwidth is around 112GB/s, which means the communication will take around 3.4 ms. This is pure communication time, while the computation time is around 5 ms based on Firework's experiments, acturally decreasing a lot at the critical batch size. So the communication overhead is significant, and it can make MoE models less efficient than dense models in the decode stage.

> How we compute the critical batch size?
> The equation is :
> $$ \text{critical batch size} = \frac{\text{FLOP/s}}{\text{Memory bandwidth(TB/s)}} $$
> For a matmul operation, assume the weight matrix is of size $d_{in} \times d_{out}$, and the input matrix is of size $b \times d_{in}$, then the total FLOP is $2 \cdot b \cdot d_{in} \cdot d_{out}$, and the total memory access is $b \cdot d_{in} + d_{in} \cdot d_{out}$. So the critical batch size can be calculated as:
> $$\frac{d_{in} \times d_{out} \times \text{BytesPerElement}}{\text{Memory bandwidth(TB/s)}} =\frac{d_{in} \times d_{out} \times 2}{\text{FLOP/s}}$$
> So on H100, the B is around 300 for a matmul operation with $d_{in} = d_{out} = 8192$.

## MoE models tend to be shallower and wider than dense models

This is a fact that is observed in practice that MoE models tend to have fewer layers and bigger d than dense models. So the serial computation time of MoE models can be smaller than dense models. This is obvious and I won't go into details here.

## At the fixed model depth, MoE models have fewer communication than dense models
The amount of network communication for the feedforward blocks needed per processed or generated token scales with the product of the model dimension, the model depth, and the number of active experts.
> The equation is:
> $$\text{Communication} \propto d \times L \times E$$
> where $d$ is the model dimension, $L$ is the model depth, and $E$ is the number of active experts.
> Each all-reduce operation's communication cost is $d$, $L$ layers and $E$ active experts will lead to a total communication cost of $d \times L \times E$.
> Take GPT-4 as an example, it has 16 experts, and each layer's parameter number $\propto d^2$, if the dense model want to achieve the same parameter number, it needs to :
$$ d_{dense}^2 = 16 \times d_{moe}^2  \Rightarrow d_{dense} = 4 \times d_{moe} $$
So the communication cost of the dense model will be:
$$\text{Communication}_d \propto 4d \times L \times 1 = 2 \times \text{Communication}_m$$
So the dense model will have 2 times more communication than the MoE model at the same model depth. 
> Generalize to the general case, set the number of experts to be $E$, then the active experts is $k$, so the communication cost fraction of the MoE model compared to the dense model will be:
$$\frac{\text{Communication}_m}{\text{Communication}_d} = \frac{k}{\sqrt{E}}$$
So when $k < \sqrt{E}$, the MoE model will have less communication than the dense model.

In practice, MoE is usually shallower and wider, which increases their communication advantage further over dense models of the same size.

## MoE models tend to have smaller attention blocks
The majority of the parameters in MoE are housed within the experts which are sparsely activated. So we can use smaller $d$.

Set the model's total parameter number to be $M$, the number of experts to be $N$:
$$ M \approx N \times d^2  \Rightarrow d \approx \sqrt{\frac{M}{N}} $$
So the model dimension of MoE models is smaller than dense models, which can lead to smaller attention blocks. Besides the kv cache of single token is also smaller, which can reduce the communication and computation of the attention blocks.


## Conclusion
Mixture-of-experts models are generally cheaper to serve for inference compared to dense models, but except in prefill this is not directly because they have a smaller number of active parameters.

They also tend to be shallower and wider than dense models, which can contribute to faster inference times. Additionally, MoE models tend to have smaller attention blocks, which can reduce the communication and computation of the attention blocks.