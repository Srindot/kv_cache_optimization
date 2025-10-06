#!/usr/bin/env python3
"""
Main execution script for KV Cache optimization experiments.

This script provides a centralized way to run all optimization experiments
and generate comparison results.
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.runner import run_full_comparison, ExperimentRunner, ExperimentConfig
from src.optimizers import create_optimizer
from src.plotting import ResultsPlotter, create_summary_report


def main():
    parser = argparse.ArgumentParser(description='Run KV Cache Optimization Experiments')
    parser.add_argument('--model', default='gpt2-large', help='Model name to use')
    parser.add_argument('--length', type=int, default=512, help='Generation length')
    parser.add_argument('--results-dir', default='./results', help='Results directory')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['baseline', 'attention_sink', 'minicache', 'vllm', 'transformers', 
                               'h2o', 'pyramidkv', 'quantization', 'quantization_8bit', 'quantization_4bit',
                               'sliding_window', 'gqa', 'lorc', 'lorc_8bit', 'lorc_fp16',
                               'scheduling_fcfs', 'scheduling_prefix', 'smoothquant', 'smoothquant_05', 'smoothquant_08',
                               'non_transformer_rwkv', 'non_transformer_mamba', 'architecture_alteration',
                               'arch_alt_10x', 'arch_alt_20x', 'all'],
                       default=['all'], help='Experiments to run')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    parser.add_argument('--no-plots', action='store_true', help='Do not generate plots')
    
    args = parser.parse_args()
    
    print("KV Cache Optimization Experiment Runner")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Generation Length: {args.length}")
    print(f"Results Directory: {args.results_dir}")
    print(f"Experiments: {args.experiments}")
    print()
    
    # Create configuration
    config = ExperimentConfig(
        model_name=args.model,
        generation_length=args.length,
        save_results=not args.no_save,
        results_dir=args.results_dir
    )
    
    # Create runner
    runner = ExperimentRunner(config)
    
    # Determine which experiments to run
    if 'all' in args.experiments:
        print("Running all available experiments...")
        results = run_full_comparison(
            model_name=args.model,
            generation_length=args.length,
            save_results=not args.no_save,
            results_dir=args.results_dir
        )
    else:
        print(f"Running selected experiments: {args.experiments}")
        
        # Create optimizers for selected experiments
        optimizers = []
        experiment_names = []
        
        for exp in args.experiments:
            if exp == 'baseline':
                optimizers.append(create_optimizer('baseline', device=config.device))
                experiment_names.append('Baseline')
            elif exp == 'attention_sink':
                optimizers.append(create_optimizer('attention_sink', window_size=128, sink_size=4, device=config.device))
                experiment_names.append('Attention Sink')
            elif exp == 'minicache':
                optimizers.append(create_optimizer('minicache', merge_start_layer=18, device=config.device))
                experiment_names.append('MiniCache')
            elif exp == 'vllm':
                optimizers.append(create_optimizer('vllm', device=config.device))
                experiment_names.append('vLLM Optimized')
            elif exp == 'transformers':
                optimizers.append(create_optimizer('transformers', device=config.device))
                experiment_names.append('Transformers Baseline')
            elif exp == 'h2o':
                optimizers.append(create_optimizer('h2o', ratio=0.1, device=config.device))
                experiment_names.append('H2O Cache')
            elif exp == 'pyramidkv':
                optimizers.append(create_optimizer('pyramidkv', compression_ratios=[1.0, 0.8, 0.6, 0.4, 0.2], device=config.device))
                experiment_names.append('PyramidKV')
            elif exp == 'quantization':
                optimizers.append(create_optimizer('quantization', bit_width=8, device=config.device))
                experiment_names.append('Quantization (8-bit)')
            elif exp == 'quantization_8bit':
                optimizers.append(create_optimizer('quantization_8bit', device=config.device))
                experiment_names.append('Quantization (8-bit)')
            elif exp == 'quantization_4bit':
                optimizers.append(create_optimizer('quantization_4bit', device=config.device))
                experiment_names.append('Quantization (4-bit)')
            elif exp == 'sliding_window':
                optimizers.append(create_optimizer('sliding_window', window_size=128, device=config.device))
                experiment_names.append('Sliding Window')
            elif exp == 'gqa':
                optimizers.append(create_optimizer('gqa', num_groups=4, device=config.device))
                experiment_names.append('Grouped Query Attention')
            elif exp == 'lorc':
                optimizers.append(create_optimizer('lorc', device=config.device))
                experiment_names.append('LORC (8-bit)')
            elif exp == 'lorc_8bit':
                optimizers.append(create_optimizer('lorc_8bit', device=config.device))
                experiment_names.append('LORC (8-bit)')
            elif exp == 'lorc_fp16':
                optimizers.append(create_optimizer('lorc_fp16', device=config.device))
                experiment_names.append('LORC (FP16)')
            elif exp == 'scheduling_fcfs':
                optimizers.append(create_optimizer('scheduling_fcfs', device=config.device))
                experiment_names.append('Scheduling (FCFS)')
            elif exp == 'scheduling_prefix':
                optimizers.append(create_optimizer('scheduling_prefix', device=config.device))
                experiment_names.append('Scheduling (Prefix-Aware)')
            elif exp == 'smoothquant':
                optimizers.append(create_optimizer('smoothquant', device=config.device))
                experiment_names.append('SmoothQuant')
            elif exp == 'smoothquant_05':
                optimizers.append(create_optimizer('smoothquant_05', device=config.device))
                experiment_names.append('SmoothQuant (α=0.5)')
            elif exp == 'smoothquant_08':
                optimizers.append(create_optimizer('smoothquant_08', device=config.device))
                experiment_names.append('SmoothQuant (α=0.8)')
            elif exp == 'non_transformer_rwkv':
                optimizers.append(create_optimizer('non_transformer_rwkv', device=config.device))
                experiment_names.append('Non-Transformer (RWKV)')
            elif exp == 'non_transformer_mamba':
                optimizers.append(create_optimizer('non_transformer_mamba', device=config.device))
                experiment_names.append('Non-Transformer (Mamba)')
            elif exp == 'architecture_alteration':
                optimizers.append(create_optimizer('architecture_alteration', device=config.device))
                experiment_names.append('Architecture Alteration')
            elif exp == 'arch_alt_10x':
                optimizers.append(create_optimizer('arch_alt_10x', device=config.device))
                experiment_names.append('Arch Alteration (10x)')
            elif exp == 'arch_alt_20x':
                optimizers.append(create_optimizer('arch_alt_20x', device=config.device))
                experiment_names.append('Arch Alteration (20x)')
        
        # Run experiments
        results = runner.run_multiple_experiments(optimizers, experiment_names)
    
    # Generate plots if requested
    if not args.no_plots and results:
        print("\nGenerating plots...")
        plotter = ResultsPlotter()
        plot_dir = os.path.join(args.results_dir, "plots")
        plotter.create_all_plots(results, save_dir=plot_dir, show_plots=False)
        
        # Create summary report
        report_path = os.path.join(args.results_dir, "summary_report.txt")
        create_summary_report(results, report_path)
    
    print(f"\nExperiments completed! Results saved to: {args.results_dir}")
    return results


if __name__ == "__main__":
    main()