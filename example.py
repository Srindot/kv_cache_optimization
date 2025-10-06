#!/usr/bin/env python3
"""
Example script demonstrating how to use the KV Cache Optimization Framework.

This script shows basic usage patterns for running experiments and generating plots.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.runner import ExperimentRunner, ExperimentConfig
from src.optimizers import create_optimizer
from src.plotting import ResultsPlotter


def simple_example():
    """Run a simple example with baseline and attention sink optimizers."""
    print("KV Cache Optimization Framework - Simple Example")
    print("=" * 50)
    
    # Create configuration for a quick test
    config = ExperimentConfig(
        model_name="gpt2",  # Use smaller model for quick test
        generation_length=128,  # Shorter generation for speed
        save_results=True,
        results_dir="./example_results"
    )
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Create optimizers
    baseline = create_optimizer('baseline', device=config.device)
    attention_sink = create_optimizer('attention_sink', 
                                    window_size=64, 
                                    sink_size=4, 
                                    device=config.device)
    
    # Run experiments
    print("\nRunning baseline experiment...")
    baseline_results = runner.run_experiment(baseline, "Baseline")
    
    print("\nRunning attention sink experiment...")
    attention_results = runner.run_experiment(attention_sink, "Attention Sink")
    
    # Create plots
    print("\nGenerating comparison plots...")
    plotter = ResultsPlotter()
    
    results = {
        "Baseline": baseline_results,
        "Attention Sink": attention_results
    }
    
    # Generate timing comparison
    plotter.plot_timing_comparison(results, 
                                 save_path="./example_results/timing_comparison.png",
                                 show_plot=True)
    
    # Generate VRAM comparison
    plotter.plot_vram_comparison(results,
                               save_path="./example_results/vram_comparison.png", 
                               show_plot=True)
    
    print(f"\nExample completed! Results saved to: {config.results_dir}")
    
    # Print summary
    print("\nSUMMARY:")
    print("-" * 30)
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Average timing: {result.get_average_timing():.2f} ms/token")
        print(f"  Peak VRAM: {result.get_peak_vram():.2f} GB")
        print()


if __name__ == "__main__":
    simple_example()