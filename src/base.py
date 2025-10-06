"""
Base classes and utilities for KV cache optimization experiments.
"""

import torch
import time
import gc
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """Base class for all KV cache optimization strategies."""
    
    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.results = {}
        
    @abstractmethod
    def run_inference(self, model, tokenizer, generation_length: int = 512, **kwargs) -> Tuple[List[float], List[float]]:
        """
        Run inference with the optimization strategy.
        
        Returns:
            Tuple[List[float], List[float]]: (timings, vram_usage)
        """
        pass
    
    def get_vram_usage(self) -> float:
        """Returns the current GPU memory usage in GB."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated(0) / (1024**3)
        return 0.0
    
    def cleanup(self):
        """Clean up GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class ExperimentConfig:
    """Configuration class for experiments."""
    
    def __init__(self, 
                 model_name: str = "gpt2-large",
                 generation_length: int = 512,
                 device: str = "cuda",
                 batch_size: int = 1,
                 save_results: bool = True,
                 results_dir: str = "./results"):
        self.model_name = model_name
        self.generation_length = generation_length
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.save_results = save_results
        self.results_dir = results_dir

class ExperimentResults:
    """Class to store and manage experiment results."""
    
    def __init__(self, optimizer_name: str, config: ExperimentConfig):
        self.optimizer_name = optimizer_name
        self.config = config
        self.timings = []
        self.vram_usage = []
        self.metadata = {}
        
    def add_timing(self, timing: float):
        """Add a timing measurement."""
        self.timings.append(timing)
        
    def add_vram_usage(self, usage: float):
        """Add a VRAM usage measurement."""
        self.vram_usage.append(usage)
        
    def set_metadata(self, key: str, value: Any):
        """Set metadata for the experiment."""
        self.metadata[key] = value
        
    def get_average_timing(self) -> float:
        """Get average timing per token."""
        return np.mean(self.timings) if self.timings else 0.0
        
    def get_peak_vram(self) -> float:
        """Get peak VRAM usage."""
        return max(self.vram_usage) if self.vram_usage else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for saving."""
        return {
            'optimizer_name': self.optimizer_name,
            'config': {
                'model_name': self.config.model_name,
                'generation_length': self.config.generation_length,
                'device': self.config.device,
                'batch_size': self.config.batch_size
            },
            'timings': self.timings,
            'vram_usage': self.vram_usage,
            'metadata': self.metadata,
            'average_timing': self.get_average_timing(),
            'peak_vram': self.get_peak_vram()
        }