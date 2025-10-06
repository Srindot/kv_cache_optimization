"""
Experiment runner and result management utilities.
"""

import os
import json
import pickle
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import ExperimentConfig, ExperimentResults, BaseOptimizer
from .optimizers import create_optimizer


class ExperimentRunner:
    """Main class for running KV cache optimization experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.results = {}
        
    def setup_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")
        print(f"Using device: {self.config.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.config.device)
        self.model.config.use_cache = True
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model setup complete.")
        
    def run_experiment(self, optimizer: BaseOptimizer, experiment_name: Optional[str] = None) -> ExperimentResults:
        """Run a single experiment with the given optimizer."""
        if self.model is None or self.tokenizer is None:
            self.setup_model()
        
        experiment_name = experiment_name or optimizer.name
        print(f"\n{'='*50}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*50}")
        
        # Create results object
        results = ExperimentResults(experiment_name, self.config)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the optimizer
            timings, vram_usage = optimizer.run_inference(
                self.model, 
                self.tokenizer, 
                self.config.generation_length,
                model_name=self.config.model_name,
                batch_size=self.config.batch_size
            )
            
            # Store results
            results.timings = timings
            results.vram_usage = vram_usage
            results.set_metadata('total_experiment_time', time.time() - start_time)
            results.set_metadata('optimizer_config', optimizer.__dict__)
            
            print(f"\nExperiment '{experiment_name}' completed successfully!")
            print(f"Average timing: {results.get_average_timing():.2f} ms/token")
            print(f"Peak VRAM: {results.get_peak_vram():.2f} GB")
            
        except Exception as e:
            print(f"Error running experiment '{experiment_name}': {str(e)}")
            results.set_metadata('error', str(e))
        
        finally:
            # Cleanup
            optimizer.cleanup()
        
        # Store results
        self.results[experiment_name] = results
        
        # Save results if configured
        if self.config.save_results:
            self.save_results(experiment_name, results)
        
        return results
    
    def run_multiple_experiments(self, optimizers: List[BaseOptimizer], experiment_names: Optional[List[str]] = None) -> Dict[str, ExperimentResults]:
        """Run multiple experiments in sequence."""
        if experiment_names is None:
            experiment_names = [opt.name for opt in optimizers]
        
        if len(experiment_names) != len(optimizers):
            raise ValueError("Number of experiment names must match number of optimizers")
        
        print(f"\nRunning {len(optimizers)} experiments...")
        
        for optimizer, name in zip(optimizers, experiment_names):
            self.run_experiment(optimizer, name)
            
        return self.results
    
    def save_results(self, experiment_name: str, results: ExperimentResults):
        """Save experiment results to disk."""
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Save as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{experiment_name}_{timestamp}.json"
        json_path = os.path.join(self.config.results_dir, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Save as pickle for full object
        pickle_filename = f"{experiment_name}_{timestamp}.pkl"
        pickle_path = os.path.join(self.config.results_dir, pickle_filename)
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to: {json_path}")
    
    def load_results(self, filepath: str) -> ExperimentResults:
        """Load experiment results from disk."""
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Reconstruct ExperimentResults object from JSON data
            config = ExperimentConfig(**data['config'])
            results = ExperimentResults(data['optimizer_name'], config)
            results.timings = data['timings']
            results.vram_usage = data['vram_usage']
            results.metadata = data['metadata']
            return results
        else:
            raise ValueError("Unsupported file format. Use .pkl or .json")
    
    def get_comparison_data(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison data for all experiments."""
        comparison = {}
        
        for name, results in self.results.items():
            comparison[name] = {
                'average_timing': results.get_average_timing(),
                'peak_vram': results.get_peak_vram(),
                'total_tokens': len(results.timings),
                'metadata': results.metadata
            }
        
        return comparison


def create_standard_experiments(config: ExperimentConfig) -> List[BaseOptimizer]:
    """Create a standard set of optimization experiments."""
    optimizers = [
        create_optimizer('baseline', device=config.device),
        create_optimizer('attention_sink', window_size=128, sink_size=4, device=config.device),
        create_optimizer('minicache', merge_start_layer=18, device=config.device),
    ]
    
    # Add memory optimizers for smaller models
    if 'gpt2' in config.model_name.lower() and 'large' not in config.model_name.lower():
        optimizers.extend([
            create_optimizer('vllm', device=config.device),
            create_optimizer('transformers', device=config.device),
        ])
    
    return optimizers


def run_full_comparison(model_name: str = "gpt2-large", 
                       generation_length: int = 512,
                       save_results: bool = True,
                       results_dir: str = "./results") -> Dict[str, ExperimentResults]:
    """Run a full comparison of all optimization strategies."""
    
    # Create configuration
    config = ExperimentConfig(
        model_name=model_name,
        generation_length=generation_length,
        save_results=save_results,
        results_dir=results_dir
    )
    
    # Create runner
    runner = ExperimentRunner(config)
    
    # Create standard experiments
    optimizers = create_standard_experiments(config)
    
    # Run all experiments
    results = runner.run_multiple_experiments(optimizers)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    comparison = runner.get_comparison_data()
    for name, data in comparison.items():
        print(f"{name}:")
        print(f"  Average timing: {data['average_timing']:.2f} ms/token")
        print(f"  Peak VRAM: {data['peak_vram']:.2f} GB")
        print(f"  Total tokens: {data['total_tokens']}")
        print()
    
    return results