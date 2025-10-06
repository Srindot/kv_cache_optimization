"""
KV Cache Optimization Strategies Implementation
"""

import torch
import time
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseOptimizer, ExperimentResults, ExperimentConfig


class BaselineOptimizer(BaseOptimizer):
    """Baseline implementation with full KV cache."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__("Baseline", device)
    
    def run_inference(self, model, tokenizer, generation_length: int = 512, **kwargs) -> Tuple[List[float], List[float]]:
        """Run baseline inference with full KV cache."""
        print(f"\n--- Running test for strategy: '{self.name}' ---")
        
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(self.device)
        past_key_values = None
        timings = []
        vram_usage = []
        
        with torch.no_grad():
            for i in range(generation_length):
                start_time = time.perf_counter()
                outputs = model(input_ids=input_ids[:, -1:], past_key_values=past_key_values)
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                past_key_values = outputs.past_key_values
                end_time = time.perf_counter()
                
                timings.append((end_time - start_time) * 1000)
                vram_usage.append(self.get_vram_usage())
                
                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{generation_length} tokens...")
        
        print(f"'{self.name}' test complete.")
        return timings, vram_usage


class AttentionSinkOptimizer(BaseOptimizer):
    """Attention Sink optimization strategy."""
    
    def __init__(self, window_size: int = 128, sink_size: int = 4, device: str = "cuda"):
        super().__init__("Attention Sink", device)
        self.window_size = window_size
        self.sink_size = sink_size
    
    def run_inference(self, model, tokenizer, generation_length: int = 512, **kwargs) -> Tuple[List[float], List[float]]:
        """Run inference with attention sink strategy."""
        print(f"\n--- Running test for strategy: '{self.name}' (Window={self.window_size}, Sink={self.sink_size}) ---")
        
        dummy_batch_size = kwargs.get('batch_size', 16)
        
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(self.device)
        input_ids = input_ids.expand(dummy_batch_size, -1)
        
        past_key_values = None
        timings = []
        vram_usage = []
        
        with torch.no_grad():
            for i in range(generation_length - 1):
                start_time = time.perf_counter()

                outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                input_ids = next_token_id.expand(dummy_batch_size, -1)
                past_key_values = outputs.past_key_values

                # KV Cache Eviction Logic
                if past_key_values is not None:
                    current_cache_size = past_key_values[0][0].shape[2]
                    
                    if current_cache_size > self.window_size:
                        past_key_values = tuple(
                            (
                                torch.cat([layer_past[0][:, :, :self.sink_size, :], 
                                          layer_past[0][:, :, -(self.window_size - self.sink_size):, :]], dim=2),
                                torch.cat([layer_past[1][:, :, :self.sink_size, :], 
                                          layer_past[1][:, :, -(self.window_size - self.sink_size):, :]], dim=2)
                            ) for layer_past in past_key_values
                        )

                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)
                vram_usage.append(self.get_vram_usage())

                if (i + 1) % 50 == 0:
                    print(f"Generated {i + 1}/{generation_length-1} tokens...")
        
        print(f"'{self.name}' test complete.")
        return timings, vram_usage


class MiniCacheOptimizer(BaseOptimizer):
    """MiniCache optimization strategy."""
    
    def __init__(self, merge_start_layer: int = 18, merge_end_layer: Optional[int] = None, device: str = "cuda"):
        super().__init__("MiniCache", device)
        self.merge_start_layer = merge_start_layer
        self.merge_end_layer = merge_end_layer
    
    def apply_minicache(self, pkv, merge_start_layer: int, merge_end_layer: int):
        """Apply MiniCache logic to merge layers."""
        shared_cache = pkv[merge_start_layer]
        new_pkv_list = list(pkv)
        
        for i in range(merge_start_layer + 1, merge_end_layer):
            new_pkv_list[i] = shared_cache
            
        return tuple(new_pkv_list)
    
    def run_inference(self, model, tokenizer, generation_length: int = 512, **kwargs) -> Tuple[List[float], List[float]]:
        """Run inference with MiniCache strategy."""
        merge_end = self.merge_end_layer or model.config.num_hidden_layers
        
        print(f"\n--- Running test for strategy: '{self.name}' (Layers {self.merge_start_layer}-{merge_end-1} Merged) ---")
        
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(self.device)
        past_key_values = None
        timings = []
        vram_usage = []

        with torch.no_grad():
            for i in range(generation_length):
                start_time = time.perf_counter()
                outputs = model(input_ids=input_ids[:, -1:], past_key_values=past_key_values)
                
                pkv = outputs.past_key_values
                if pkv is not None:
                    past_key_values = self.apply_minicache(pkv, self.merge_start_layer, merge_end)
                
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                end_time = time.perf_counter()

                timings.append((end_time - start_time) * 1000)
                vram_usage.append(self.get_vram_usage())
                
                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{generation_length} tokens...")
                    
        print(f"'{self.name}' test complete.")
        return timings, vram_usage


class MemoryOptimizer(BaseOptimizer):
    """Memory optimization comparison between vLLM and Transformers."""
    
    def __init__(self, use_vllm: bool = True, device: str = "cuda"):
        name = "vLLM Optimized" if use_vllm else "Transformers Baseline"
        super().__init__(name, device)
        self.use_vllm = use_vllm
    
    def run_inference(self, model, tokenizer, generation_length: int = 128, **kwargs) -> Tuple[List[float], List[float]]:
        """Run inference comparing memory optimization approaches."""
        prompts = kwargs.get('prompts', [
            "The weather today is",
            "In a shocking turn of events, scientists have discovered a new species of deep-sea fish that glows in the dark. This discovery",
            "To make the perfect cup of coffee, you must first",
            "The history of the Roman Empire is a complex tapestry woven with threads of conquest, innovation, and betrayal. It began",
            "Hello!",
        ])
        
        if self.use_vllm:
            return self._run_vllm_inference(prompts, generation_length, **kwargs)
        else:
            return self._run_transformers_inference(model, tokenizer, prompts, generation_length, **kwargs)
    
    def _run_vllm_inference(self, prompts: List[str], generation_length: int, **kwargs) -> Tuple[List[float], List[float]]:
        """Run optimized inference with vLLM."""
        print(f"--- Running test for strategy: '{self.name}' ---")
        
        if self.device == "cpu":
            print("vLLM requires a GPU. Skipping this test.")
            return [0], [0]

        try:
            from vllm import LLM, SamplingParams
            
            llm = LLM(
                model=kwargs.get('model_name', 'gpt2'),
                trust_remote_code=True,
                gpu_memory_utilization=0.6
            )
            sampling_params = SamplingParams(temperature=0.7, max_tokens=generation_length)

            start_time = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            end_time = time.perf_counter()

            total_time = end_time - start_time
            total_tokens_generated = sum(len(output.outputs[0].token_ids) for output in outputs)
            throughput = total_tokens_generated / total_time

            print(f"Total time taken: {total_time:.2f} seconds")
            print(f"Total tokens generated: {total_tokens_generated}")
            print(f"Throughput: {throughput:.2f} tokens/second")

            del llm
            self.cleanup()
            
            # Return single values as lists for consistency
            return [total_time * 1000], [self.get_vram_usage()]
            
        except ImportError:
            print("vLLM not available. Install with: pip install vllm")
            return [0], [0]
    
    def _run_transformers_inference(self, model, tokenizer, prompts: List[str], generation_length: int, **kwargs) -> Tuple[List[float], List[float]]:
        """Run non-optimized inference with Transformers."""
        print(f"--- Running test for strategy: '{self.name}' ---")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        start_time = time.perf_counter()
        generated_outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=generation_length,
            attention_mask=inputs.attention_mask
        )
        end_time = time.perf_counter()

        total_time = end_time - start_time
        num_input_tokens = inputs.input_ids.shape[1]
        total_tokens_generated = sum(len(output) for output in generated_outputs) - (len(prompts) * num_input_tokens)
        throughput = total_tokens_generated / total_time

        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Total tokens generated: {total_tokens_generated}")
        print(f"Throughput: {throughput:.2f} tokens/second")

        # Return single values as lists for consistency
        return [total_time * 1000], [self.get_vram_usage()]


# Factory function to create optimizers
def create_optimizer(optimizer_type: str, **kwargs) -> BaseOptimizer:
    """Factory function to create optimizers."""
    optimizers = {
        'baseline': BaselineOptimizer,
        'attention_sink': AttentionSinkOptimizer,
        'minicache': MiniCacheOptimizer,
        'vllm': lambda **k: MemoryOptimizer(use_vllm=True, **k),
        'transformers': lambda **k: MemoryOptimizer(use_vllm=False, **k),
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizers[optimizer_type](**kwargs)