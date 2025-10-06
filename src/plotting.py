"""
Plotting and visualization utilities for KV cache optimization results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Any
import os

from .base import ExperimentResults


class ResultsPlotter:
    """Class for creating visualizations of experiment results."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: tuple = (12, 8)):
        """Initialize plotter with style settings."""
        plt.style.use(style) if style in plt.style.available else None
        self.figsize = figsize
        self.colors = ['#d9534f', '#5cb85c', '#5bc0de', '#f0ad4e', '#d9534f', '#5cb85c']
    
    def plot_timing_comparison(self, results: Dict[str, ExperimentResults], 
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
        """Plot timing comparison across all experiments."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (name, result) in enumerate(results.items()):
            if result.timings:
                ax.plot(result.timings, label=name, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_xlabel('Generated Token Number (Sequence Length)')
        ax.set_ylabel('Time per Token (ms)')
        ax.set_title('KV Cache Performance: Timing Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timing comparison plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_vram_comparison(self, results: Dict[str, ExperimentResults], 
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
        """Plot VRAM usage comparison across all experiments."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (name, result) in enumerate(results.items()):
            if result.vram_usage:
                ax.plot(result.vram_usage, label=name, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_xlabel('Generated Token Number (Sequence Length)')
        ax.set_ylabel('VRAM Usage (GB)')
        ax.set_title('KV Cache Performance: VRAM Usage Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"VRAM comparison plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_combined_analysis(self, results: Dict[str, ExperimentResults], 
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
        """Plot combined timing and VRAM analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Timing subplot
        for i, (name, result) in enumerate(results.items()):
            if result.timings:
                ax1.plot(result.timings, label=name, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax1.set_xlabel('Generated Token Number (Sequence Length)')
        ax1.set_ylabel('Time per Token (ms)')
        ax1.set_title('Latency Analysis: KV Cache Optimization Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # VRAM subplot
        for i, (name, result) in enumerate(results.items()):
            if result.vram_usage:
                ax2.plot(result.vram_usage, label=name, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax2.set_xlabel('Generated Token Number (Sequence Length)')
        ax2.set_ylabel('VRAM Usage (GB)')
        ax2.set_title('VRAM Usage: KV Cache Optimization Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined analysis plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_performance_bars(self, results: Dict[str, ExperimentResults], 
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> None:
        """Plot bar chart comparing average performance metrics."""
        names = list(results.keys())
        avg_timings = [result.get_average_timing() for result in results.values()]
        peak_vrams = [result.get_peak_vram() for result in results.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1] * 0.8))
        
        # Average timing bar chart
        bars1 = ax1.bar(names, avg_timings, color=self.colors[:len(names)])
        ax1.set_ylabel('Average Time per Token (ms)')
        ax1.set_title('Average Timing Performance')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Peak VRAM bar chart
        bars2 = ax2.bar(names, peak_vrams, color=self.colors[:len(names)])
        ax2.set_ylabel('Peak VRAM Usage (GB)')
        ax2.set_title('Peak VRAM Usage')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance bars plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_throughput_comparison(self, results: Dict[str, ExperimentResults], 
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True) -> None:
        """Plot throughput comparison for memory optimization experiments."""
        # Filter for memory optimization results
        memory_results = {name: result for name, result in results.items() 
                         if 'vLLM' in name or 'Transformers' in name}
        
        if len(memory_results) < 2:
            print("Not enough memory optimization results for throughput comparison")
            return
        
        names = list(memory_results.keys())
        # Calculate throughput as tokens/second
        throughputs = []
        for result in memory_results.values():
            if result.timings and len(result.timings) > 0:
                # For memory optimizers, timing is total time, so throughput = tokens/time
                total_time_seconds = result.timings[0] / 1000  # Convert ms to seconds
                tokens_generated = result.metadata.get('total_tokens', len(result.timings))
                throughput = tokens_generated / total_time_seconds if total_time_seconds > 0 else 0
                throughputs.append(throughput)
            else:
                throughputs.append(0)
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(names, throughputs, color=self.colors[:len(names)])
        plt.ylabel('Throughput (Tokens per Second)')
        plt.title('Inference Performance Comparison: Memory Optimization')
        plt.ylim(0, max(throughputs) * 1.2 if throughputs else 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Calculate and display speedup if we have two results
        if len(throughputs) == 2 and throughputs[0] > 0:
            speedup = throughputs[1] / throughputs[0]
            plt.text(0.95, 0.90, f'Speedup: {speedup:.2f}x',
                    transform=plt.gca().transAxes,
                    fontsize=14,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Throughput comparison plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_all_plots(self, results: Dict[str, ExperimentResults], 
                        save_dir: Optional[str] = None,
                        show_plots: bool = True) -> None:
        """Create all available plots for the results."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Timing comparison
        timing_path = os.path.join(save_dir, "timing_comparison.png") if save_dir else None
        self.plot_timing_comparison(results, timing_path, show_plots)
        
        # VRAM comparison
        vram_path = os.path.join(save_dir, "vram_comparison.png") if save_dir else None
        self.plot_vram_comparison(results, vram_path, show_plots)
        
        # Combined analysis
        combined_path = os.path.join(save_dir, "combined_analysis.png") if save_dir else None
        self.plot_combined_analysis(results, combined_path, show_plots)
        
        # Performance bars
        bars_path = os.path.join(save_dir, "performance_bars.png") if save_dir else None
        self.plot_performance_bars(results, bars_path, show_plots)
        
        # Throughput comparison (if applicable)
        throughput_path = os.path.join(save_dir, "throughput_comparison.png") if save_dir else None
        self.plot_throughput_comparison(results, throughput_path, show_plots)


def create_summary_report(results: Dict[str, ExperimentResults], 
                         save_path: Optional[str] = None) -> str:
    """Create a text summary report of all experiments."""
    report = []
    report.append("=" * 60)
    report.append("KV CACHE OPTIMIZATION EXPERIMENT SUMMARY")
    report.append("=" * 60)
    report.append("")
    
    for name, result in results.items():
        report.append(f"Experiment: {name}")
        report.append("-" * 40)
        report.append(f"Model: {result.config.model_name}")
        report.append(f"Generation Length: {result.config.generation_length}")
        report.append(f"Device: {result.config.device}")
        report.append(f"Average Timing: {result.get_average_timing():.2f} ms/token")
        report.append(f"Peak VRAM: {result.get_peak_vram():.2f} GB")
        report.append(f"Total Tokens: {len(result.timings)}")
        
        if result.metadata:
            report.append("Metadata:")
            for key, value in result.metadata.items():
                if key != 'optimizer_config':  # Skip complex nested objects
                    report.append(f"  {key}: {value}")
        
        report.append("")
    
    # Calculate speedups relative to baseline
    baseline_result = results.get('Baseline') or results.get('baseline')
    if baseline_result:
        baseline_time = baseline_result.get_average_timing()
        baseline_vram = baseline_result.get_peak_vram()
        
        report.append("PERFORMANCE COMPARISON (vs Baseline)")
        report.append("-" * 40)
        
        for name, result in results.items():
            if name.lower() != 'baseline':
                time_speedup = baseline_time / result.get_average_timing() if result.get_average_timing() > 0 else 0
                vram_reduction = (baseline_vram - result.get_peak_vram()) / baseline_vram * 100 if baseline_vram > 0 else 0
                
                report.append(f"{name}:")
                report.append(f"  Timing Speedup: {time_speedup:.2f}x")
                report.append(f"  VRAM Reduction: {vram_reduction:.1f}%")
                report.append("")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Summary report saved to: {save_path}")
    
    return report_text