"""
SmartCache Visualization Module
================================
Generate plots and visualizations for cache optimization results:
- Pareto frontiers (size vs. miss rate trade-offs)
- Performance comparisons (AI vs. baselines)
- Optimization convergence curves
- Heatmaps of design space exploration
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import json
import seaborn as sns

# Set nice styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SmartCacheVisualizer:
    """
    Visualization toolkit for SmartCache experiments.
    
    Generates publication-quality plots demonstrating:
    1. Pareto frontiers showing size/performance trade-offs
    2. AI vs. baseline performance comparisons
    3. Convergence behavior of Bayesian optimization
    4. Design space exploration patterns
    """
    
    def __init__(self, figsize: tuple = (10, 6), dpi: int = 100):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size (width, height)
        dpi : int
            Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_pareto_frontier(self, pareto_points: List[Dict],
                            title: str = "Pareto Frontier: Size vs. Miss Rate",
                            save_path: Optional[str] = None):
        """
        Plot Pareto frontier showing trade-off between cache size and miss rate.
        
        The Pareto frontier represents the "efficient frontier" where you
        cannot improve one objective (miss rate) without worsening another
        (cache size). This is crucial for design decisions:
        - Points on the frontier are Pareto-optimal
        - Points below/right are dominated (worse in all objectives)
        
        Parameters:
        -----------
        pareto_points : List[Dict]
            List of Pareto-optimal configurations
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if not pareto_points:
            print("WARNING: No Pareto points to plot")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract data
        sizes = [p['cache_size'] / 1024 for p in pareto_points]  # Convert to KB
        miss_rates = [p['miss_rate'] * 100 for p in pareto_points]  # Convert to %
        
        # Plot Pareto curve
        ax.plot(sizes, miss_rates, 'o-', linewidth=2, markersize=8,
               label='Pareto Frontier', color='#2E86AB')
        
        # Annotate points
        for i, (s, m, p) in enumerate(zip(sizes, miss_rates, pareto_points)):
            if i % max(1, len(pareto_points) // 5) == 0:  # Annotate every ~5th point
                ax.annotate(f"{int(s)}KB\n{p['associativity']}-way",
                          xy=(s, m), xytext=(10, -10),
                          textcoords='offset points',
                          fontsize=8, alpha=0.7,
                          bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='yellow', alpha=0.3))
        
        ax.set_xlabel('Cache Size (KB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Miss Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved Pareto frontier to {save_path}")
        
        plt.show()
    
    def plot_multiple_pareto_frontiers(self, 
                                      workload_results: Dict[str, List[Dict]],
                                      save_path: Optional[str] = None):
        """
        Plot Pareto frontiers for multiple workloads on same axes.
        
        This visualization shows how optimal configurations differ
        across workloads - demonstrating why "one-size-fits-all" fails.
        
        Parameters:
        -----------
        workload_results : dict
            Dict mapping workload names to their Pareto points
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(workload_results)))
        
        for (workload_name, pareto_points), color in zip(workload_results.items(), colors):
            if not pareto_points:
                continue
            
            sizes = [p['cache_size'] / 1024 for p in pareto_points]
            miss_rates = [p['miss_rate'] * 100 for p in pareto_points]
            
            ax.plot(sizes, miss_rates, 'o-', linewidth=2, markersize=6,
                   label=workload_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Cache Size (KB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Miss Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Pareto Frontiers Across Workloads', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved multi-Pareto plot to {save_path}")
        
        plt.show()
    
    def plot_ai_vs_baseline_comparison(self, comparison_results: Dict,
                                      save_path: Optional[str] = None):
        """
        Bar chart comparing AI-optimized vs. best baseline for each workload.
        
        This is the key plot demonstrating AI superiority over static designs.
        
        Parameters:
        -----------
        comparison_results : dict
            Results from experiment_framework.compare_performance()
        save_path : str, optional
            Path to save figure
        """
        if not comparison_results:
            print("WARNING: No comparison results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=self.dpi)
        
        workloads = list(comparison_results.keys())
        ai_miss_rates = [comparison_results[w]['ai_miss_rate'] * 100 
                        for w in workloads]
        baseline_miss_rates = [comparison_results[w]['best_baseline_miss_rate'] * 100
                              for w in workloads]
        improvements = [comparison_results[w]['relative_improvement_pct']
                       for w in workloads]
        
        x = np.arange(len(workloads))
        width = 0.35
        
        # Plot 1: Miss rates comparison
        bars1 = ax1.bar(x - width/2, baseline_miss_rates, width,
                       label='Best Baseline', color='#E63946', alpha=0.8)
        bars2 = ax1.bar(x + width/2, ai_miss_rates, width,
                       label='AI-Optimized', color='#06A77D', alpha=0.8)
        
        ax1.set_xlabel('Workload', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Miss Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('AI-Optimized vs. Baseline: Miss Rate Comparison',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(workloads, rotation=45, ha='right')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Improvement percentage
        colors = ['#06A77D' if imp > 0 else '#E63946' for imp in improvements]
        bars3 = ax2.bar(x, improvements, color=colors, alpha=0.8)
        
        ax2.set_xlabel('Workload', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Relative Improvement of AI over Best Baseline',
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(workloads, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        plt.show()
    
    def plot_optimization_convergence(self, optimization_history: List[Dict],
                                     title: str = "Bayesian Optimization Convergence",
                                     save_path: Optional[str] = None):
        """
        Plot convergence of Bayesian optimization over iterations.
        
        Shows how the optimizer learns and improves over time:
        - Blue line: Best miss rate found so far (cumulative minimum)
        - Orange points: Miss rate at each evaluation
        
        Demonstrates the advantage of intelligent search over random search.
        
        Parameters:
        -----------
        optimization_history : List[Dict]
            History of evaluated configurations
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if not optimization_history:
            print("WARNING: No history to plot")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        iterations = list(range(1, len(optimization_history) + 1))
        miss_rates = [h['miss_rate'] * 100 for h in optimization_history]
        
        # Calculate cumulative minimum (best so far)
        best_so_far = []
        current_best = float('inf')
        for mr in miss_rates:
            current_best = min(current_best, mr)
            best_so_far.append(current_best)
        
        # Plot all evaluations
        ax.scatter(iterations, miss_rates, alpha=0.3, s=30,
                  label='Evaluations', color='#F77F00')
        
        # Plot best so far
        ax.plot(iterations, best_so_far, linewidth=2.5,
               label='Best Found', color='#06A77D')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Miss Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Annotate final best
        final_best = best_so_far[-1]
        ax.annotate(f'Final: {final_best:.2f}%',
                   xy=(len(iterations), final_best),
                   xytext=(-60, 20),
                   textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', 
                                  connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved convergence plot to {save_path}")
        
        plt.show()
    
    def plot_design_space_exploration(self, optimization_history: List[Dict],
                                     save_path: Optional[str] = None):
        """
        2D scatter plot showing explored cache configurations.
        
        Visualizes how Bayesian optimization explores the design space:
        - X-axis: Cache size
        - Y-axis: Miss rate
        - Color: Associativity
        - Size: Optimization iteration
        
        Shows intelligent exploration pattern vs. random search.
        
        Parameters:
        -----------
        optimization_history : List[Dict]
            History of evaluated configurations
        save_path : str, optional
            Path to save figure
        """
        if not optimization_history:
            print("WARNING: No history to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=self.dpi)
        
        sizes = [h['cache_size'] / 1024 for h in optimization_history]
        miss_rates = [h['miss_rate'] * 100 for h in optimization_history]
        assocs = [h['associativity'] for h in optimization_history]
        iterations = list(range(len(optimization_history)))
        
        # Scatter plot with multiple dimensions encoded
        scatter = ax.scatter(sizes, miss_rates, c=assocs, s=[20 + i*2 for i in iterations],
                           alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Color bar for associativity
        cbar = plt.colorbar(scatter, ax=ax, label='Associativity')
        cbar.set_label('Associativity (N-way)', fontsize=11, fontweight='bold')
        
        # Highlight best point
        best_idx = np.argmin(miss_rates)
        ax.scatter([sizes[best_idx]], [miss_rates[best_idx]], 
                  s=400, marker='*', color='red', 
                  edgecolors='black', linewidth=2,
                  label='Best Configuration', zorder=10)
        
        ax.set_xlabel('Cache Size (KB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Miss Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Design Space Exploration by Bayesian Optimization',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        
        # Add note about marker size
        ax.text(0.02, 0.98, 'Marker size ∝ iteration number',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved design space plot to {save_path}")
        
        plt.show()
    
    def create_summary_report(self, results_file: str = 'experiment_results.json',
                             output_dir: str = 'plots'):
        """
        Generate complete set of visualizations from saved results.
        
        Creates publication-ready figures for all key analyses.
        
        Parameters:
        -----------
        results_file : str
            Path to JSON results file
        output_dir : str
            Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*70)
        print("GENERATING VISUALIZATION SUMMARY REPORT")
        print("="*70)
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"\nLoaded results from {results_file}")
        print(f"   Workloads: {len(results['optimized'])}")
        print(f"   Output directory: {output_dir}\n")
        
        # Plot Pareto frontiers for each workload
        print("Generating Pareto frontier plots...")
        for workload_name, opt_result in results['optimized'].items():
            pareto_points = opt_result['pareto_frontier']
            if pareto_points:
                save_path = os.path.join(output_dir, 
                                        f'pareto_{workload_name}.png')
                self.plot_pareto_frontier(
                    pareto_points,
                    title=f'Pareto Frontier: {workload_name}',
                    save_path=save_path
                )
        
        print("\nVisualization report complete!")
        print(f"   All plots saved to {output_dir}/")
        print("="*70)


if __name__ == "__main__":
    # Demonstration with synthetic data
    print("SmartCache Visualizer - Demonstration")
    print("="*60)
    
    # Create synthetic Pareto frontier
    synthetic_pareto = [
        {'cache_size': 1024, 'block_size': 32, 'associativity': 1, 'miss_rate': 0.45},
        {'cache_size': 2048, 'block_size': 32, 'associativity': 2, 'miss_rate': 0.35},
        {'cache_size': 4096, 'block_size': 64, 'associativity': 2, 'miss_rate': 0.28},
        {'cache_size': 8192, 'block_size': 64, 'associativity': 4, 'miss_rate': 0.20},
        {'cache_size': 16384, 'block_size': 64, 'associativity': 8, 'miss_rate': 0.15},
        {'cache_size': 32768, 'block_size': 128, 'associativity': 8, 'miss_rate': 0.12},
    ]
    
    # Create synthetic optimization history
    np.random.seed(42)
    synthetic_history = []
    for i in range(50):
        synthetic_history.append({
            'cache_size': 2 ** np.random.randint(10, 16),
            'block_size': 2 ** np.random.randint(5, 8),
            'associativity': 2 ** np.random.randint(0, 4),
            'miss_rate': max(0.05, 0.5 - i * 0.008 + np.random.normal(0, 0.02))
        })
    
    # Create visualizer
    viz = SmartCacheVisualizer()
    
    # Generate demonstration plots
    print("\nGenerating demonstration plots...\n")
    
    print("1. Pareto Frontier")
    viz.plot_pareto_frontier(synthetic_pareto)
    
    print("\n2. Optimization Convergence")
    viz.plot_optimization_convergence(synthetic_history)
    
    print("\n3. Design Space Exploration")
    viz.plot_design_space_exploration(synthetic_history)
    
    print("\nDemonstration complete!")
