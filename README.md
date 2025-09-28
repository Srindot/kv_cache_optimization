# KV Cache Management for LLMs
This repository contains the course project of the course Hardware for AI, the course project aims to optimize the kv caching for a particular arch accelerator design.

# Problem Statement
Explore strategies to reduce redundant computations and improve memory utilization during long-context real-time LLM inferencing
Ref: A Survey on Large Language Model
Acceleration based on KV Cache Management

Project Summary: KV Cache Optimization for LLM Inference

Goal: To implement and evaluate one or more KV cache optimization techniques to reduce the memory footprint and accelerate the inference speed of Large Language Models, based on the strategies outlined in the survey paper.

Implementation Plan (Tiered Approach):

    Core Project (Medium Difficulty - The "Safe Net"):

        Task: Implement a Token-Level optimization.

        Method: Apply a cache eviction policy (e.g., a sliding window with an attention sink) to a standard pre-trained model from Hugging Face by modifying its generation loop.

    Success Metric: Demonstrate reduced GPU memory usage and analyze the trade-off with text generation quality.

Stretch Goal (Hard Difficulty):

    Task: Implement a Model-Level optimization if Tier 1 is completed.

    Method: Build a small "toy" transformer from scratch, train it with standard Multi-Head Attention, then implement Grouped-Query Attention (GQA), and retrain to compare performance.

    Success Metric: Show a significant reduction in KV cache size and faster inference speed, with minimal impact on the model's performance (e.g., training loss).

Expert Goal (Really Hard Difficulty):

    Task: Investigate System-Level optimization if Tier 2 is completed.

    Method: Due to the high complexity requiring CUDA/C++ expertise, the focus will be on research and analysis. Deeply study an I/O-aware kernel like 

    FlashAttention and include a detailed breakdown of its mechanism and impact in the final project report, rather than attempting a full implementation.


# Papers Content & Ideas for the topic 
Of course. I can certainly help you understand this survey paper on KV cache management for your course project on "Hardware for AI." Let's break it down into two parts: first, I'll explain what the paper is about, and then I'll suggest some practical ideas you could implement for your project.

### Part 1: Explanation of the Survey Paper

The paper you've provided, "A Survey on Large Language Model Acceleration based on KV Cache Management," is a comprehensive overview of techniques used to make Large Language Models (LLMs) faster and more memory-efficient during inference (the process of generating text). The central focus is on managing the **Key-Value (KV) cache**.

#### What is the KV Cache and Why is it Important?

*   **LLM Inference Process:** LLMs like GPT generate text one token (word or sub-word) at a time, in a process called auto-regressive generation. To generate the next token, the model needs to consider all the tokens that came before it.
*   **The Bottleneck:** This process can be slow and memory-intensive because, for each new token, the model has to re-calculate intermediate values (called Keys and Values) for all previous tokens in the sequence. This computational complexity grows quadratically with the sequence length.
*   **The Solution (KV Cache):** The KV cache is an optimisation technique that stores these Key and Value matrices from previous steps. By reusing these cached values, the model avoids redundant computations and only needs to calculate the K and V for the newest token. This dramatically speeds up inference.
*   **The New Problem:** While the KV cache boosts speed, it introduces a new challenge: **it consumes a lot of memory**. The size of the cache grows linearly with the sequence length and the number of layers in the model, which can quickly exceed the memory available on a GPU, especially for long contexts.

#### What the Paper Covers: A Taxonomy of Solutions

The survey organises existing solutions for managing the KV cache into a clear taxonomy with three main levels, as illustrated in its detailed diagrams (Fig. 2):

1.  **Token-Level Optimisation:** These techniques focus on the individual tokens within the cache without changing the model's architecture.
    *   **KV Cache Selection:** Deciding which tokens are most important to keep in the cache and which to discard (evict). This can be **static** (decided once after the prompt is processed) or **dynamic** (continuously updated during generation). An important observation is that attention is often sparse, meaning only a few "heavy-hitter" tokens are truly important.
    *   **KV Cache Budget Allocation:** Instead of treating all layers or attention heads equally, this method allocates more cache memory to more important parts of the model (e.g., lower layers or specific "retrieval heads").
    *   **KV Cache Merging:** Combining similar KV pairs to reduce redundancy, either within the same layer (**intra-layer**) or across different layers (**cross-layer**).
    *   **KV Cache Quantization:** This is a very popular technique that reduces the memory footprint by converting the cached values from high-precision numbers (like Float16) to lower-precision integers (like INT8 or INT4). A key challenge here is handling "outliers"—extreme values that can cause significant performance degradation when quantized.
    *   **KV Cache Low-Rank Decomposition:** Compressing the KV cache by assuming the matrices have a low-rank structure, meaning their essential information can be captured in a much smaller form using techniques like Singular Value Decomposition (SVD).

2.  **Model-Level Optimisation:** These methods involve changing the LLM's architecture to make KV caching more efficient, often requiring model retraining.
    *   **Attention Grouping and Sharing:** Techniques like Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) have multiple query heads share the same Key and Value heads, drastically reducing the size of the KV cache. This can also be done across layers.
    *   **Architecture Alteration:** Designing new attention mechanisms or adding components (like a small encoder) to manage context more efficiently.
    *   **Non-Transformer Architectures:** Exploring entirely new architectures like Mamba or RWKV that don't rely on the traditional attention mechanism and thus don't have the same KV cache problem.

3.  **System-Level Optimisation:** These are low-level optimisations related to how memory, scheduling, and hardware are managed.
    *   **Memory Management:** Using advanced techniques inspired by operating systems, like **PagedAttention** (used in the vLLM system), which breaks the KV cache into non-contiguous blocks to avoid memory fragmentation.
    *   **Scheduling:** Intelligently scheduling user requests to maximise cache reuse, for example, by grouping requests that share the same initial prompt (prefix sharing).
    *   **Hardware-Aware Design:** Optimising for specific hardware. This includes efficient data movement between GPU, CPU, and even SSDs, and designing custom GPU kernels (like in **FlashAttention**) to minimise I/O between different levels of GPU memory.

### Part 2: What You Can Implement for Your Project

Given your course is "Hardware for AI," focusing on ideas that have a direct link to hardware utilisation, memory management, and computational efficiency would be most relevant. Here are some concrete project ideas derived from the paper, ranging from simpler to more complex:

1.  **Implement a Simple KV Cache Eviction Strategy (Token-Level):**
    *   **What to do:** Implement a basic LLM generation loop and add a KV cache. Then, implement a simple eviction policy when the cache hits a predefined size limit.
    *   **Ideas from the paper:**
        *   **Sliding Window:** Keep only the most recent `k` tokens (like in *StreamingLLM*). This is the simplest to implement.
        *   **"Attention Sink" + Recency:** Keep the first few tokens (the "attention sink") and the most recent `k` tokens. This is slightly more complex but very effective.
        *   **L2 Norm-based Eviction:** As suggested by *L2Compress*, identify important tokens by calculating the L2 norm of their key embeddings during the prompt processing phase and only cache those with a low L2 norm. This is a simple but clever heuristic.
    *   **How it relates to Hardware:** You can measure how different cache sizes affect GPU memory usage and inference latency (speed). This directly connects to hardware constraints.

2.  **Implement KV Cache Quantization (Token-Level):**
    *   **What to do:** Take a standard KV cache that stores values in FP16 and implement a function to quantize them to INT8 or INT4. You will also need to de-quantize them before the attention calculation.
    *   **Ideas from the paper:**
        *   **Fixed-Precision Per-Token Quantization:** As done in *ZeroQuant*, calculate a scaling factor for each token's K and V vector individually and quantize based on that.
        *   **Handling Outliers:** A simple way to handle outliers, inspired by *KVQuant*, is to not quantize the very first token (the "attention sink") or the most recent tokens, keeping them in full precision, while quantizing the rest.
    *   **How it relates to Hardware:** This is a classic hardware optimisation. You can demonstrate a **significant reduction in GPU memory usage** (e.g., 2x for INT8, 4x for INT4) and measure the trade-off in model accuracy (perplexity) and speed.

3.  **Implement a Simplified Grouped-Query Attention (GQA) (Model-Level):**
    *   **What to do:** Modify a standard multi-head attention implementation. Instead of each of the `N` heads having its own K and V projection, create `G` groups of heads, where each group shares a single K and V projection.
    *   **Ideas from the paper:** The GQA paper itself provides the core idea. You wouldn't need to train a model; you could take an existing multi-head model and simulate GQA by averaging the K/V weights of the heads within each group.
    *   **How it relates to Hardware:** This directly reduces the size of the KV cache stored on the GPU. You can show that for the same context length, a GQA-like model requires much less memory, allowing for larger batch sizes or longer sequences on the same hardware.

4.  **Simulate a Hierarchical Caching System (System-Level):**
    *   **What to do:** Implement a two-tier cache system. Keep a small, fixed-size KV cache on the "GPU" (a fast in-memory dictionary in your code) and offload less important tokens to a larger, slower "CPU" cache (another dictionary or even writing to a file to simulate latency).
    *   **Ideas from the paper:**
        *   *InfLLM* uses this CPU-GPU hierarchical approach. You can implement a retrieval mechanism where, if a token is needed for attention but is not in the GPU cache, it's fetched from the CPU cache.
        *   You can combine this with an eviction policy: evict from GPU to CPU, and from CPU to disk (permanent eviction).
    *   **How it relates to Hardware:** This project directly explores the trade-offs in a memory hierarchy (fast/small vs. slow/large), which is a fundamental concept in computer architecture and hardware design. You can measure the latency impact of a "cache miss" (having to fetch from the CPU).

For your project, I would recommend starting with **Idea 1 or 2**, as they are highly relevant, impactful, and can be implemented without needing to retrain a model. You can clearly demonstrate the hardware benefits (memory savings, speed changes) and analyse the performance trade-offs. Good luck!
