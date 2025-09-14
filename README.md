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

