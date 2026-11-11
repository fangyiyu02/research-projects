"""
SmartCache Experiment Framework
================================
Orchestrates experiments to demonstrate AI-driven cache optimization
outperforms traditional "one-size-fits-all" approaches.

Key Components:
1. Baseline configurations (static designs)
2. AI-optimized configurations (adaptive designs)
3. Performance comparison and analysis
4. Pareto frontier visualization
"""

import time
import json
from typing import Dict, List, Tuple
import numpy as np
from cache_simulator import CacheSimulator, evaluate_cache_config
from trace_generator import TraceGenerator
from ai_optimizer import BayesianCacheOptimizer


class SmartCacheExperiment:
    """
    Comprehensive experimental framework for cache optimization.
    
    Experimental Methodology:
    1. Define baseline "one-size-fits-all" configurations
    2. Generate realistic workload traces
    3. Evaluate baselines on all workloads
    4. Run AI optimization for each workload
    5. Compare AI-optimized vs. baseline performance
    6. Analyze Pareto frontiers
    """
    
    def __init__(self, max_cache_size: int = 65536, seed: int = 42):
        """
        Initialize experiment framework.
        
        Parameters:
        -----------
        max_cache_size : int
            Maximum cache size constraint (bytes)
        seed : int
            Random seed for reproducibility
        """
        self.max_cache_size = max_cache_size
        self.seed = seed
        self.trace_generator = TraceGenerator(seed=seed)
        
        # Results storage
        self.baseline_results = {}
        self.optimized_results = {}
        self.workloads = {}
        
    def define_baselines(self) -> Dict[str, Dict[str, int]]:
        """
        Define standard "one-size-fits-all" cache configurations.
        
        These represent typical cache designs that might be used
        without workload-specific optimization:
        
        1. Small & Fast: Low latency, good for small working sets
        2. Balanced: Middle-ground design
        3. Large & Associative: High capacity, good for diverse workloads
        
        Returns:
        --------
        baselines : dict
            Dictionary of baseline configurations
        """
        baselines = {
            'small_direct': {
                'cache_size': 4096,    # 4 KB
                'block_size': 32,      # 32 B
                'associativity': 1,    # Direct-mapped
                'description': 'Small, direct-mapped (low latency)'
            },
            'balanced': {
                'cache_size': 16384,   # 16 KB
                'block_size': 64,      # 64 B
                'associativity': 4,    # 4-way
                'description': 'Balanced design (typical L1)'
            },
            'large_assoc': {
                'cache_size': 32768,   # 32 KB
                'block_size': 64,      # 64 B
                'associativity': 8,    # 8-way
                'description': 'Large, highly associative (low conflict misses)'
            },
            'max_capacity': {
                'cache_size': self.max_cache_size,  # Max size
                'block_size': 128,     # 128 B
                'associativity': 16,   # 16-way
                'description': 'Maximum capacity (at size limit)'
            }
        }
        
        return baselines
    
    def generate_workloads(self, workload_types: List[str] = None) -> Dict[str, List[int]]:
        """
        Generate suite of workload traces.
        
        Parameters:
        -----------
        workload_types : List[str], optional
            Specific workloads to generate. If None, use default suite.
            
        Returns:
        --------
        workloads : dict
            Dictionary mapping workload names to traces
        """
        if workload_types is None:
            # Default comprehensive suite
            workloads = {
                'matmul_32': self.trace_generator.matrix_multiplication_trace(32),
                'matmul_64': self.trace_generator.matrix_multiplication_trace(64),
                'sort_1k': self.trace_generator.quicksort_trace(1000),
                'sort_5k': self.trace_generator.quicksort_trace(5000),
                'sequential': self.trace_generator.sequential_scan_trace(5000),
                'random': self.trace_generator.random_access_trace(5000, 5000),
                'stride_8': self.trace_generator.strided_access_trace(5000, stride=8),
                'mixed': self.trace_generator.mixed_workload_trace(5000)
            }
        else:
            workloads = {}
            for wl_type in workload_types:
                if wl_type.startswith('matmul'):
                    size = int(wl_type.split('_')[1])
                    workloads[wl_type] = self.trace_generator.matrix_multiplication_trace(size)
                elif wl_type.startswith('sort'):
                    size = int(wl_type.split('_')[1])
                    workloads[wl_type] = self.trace_generator.quicksort_trace(size)
                elif wl_type == 'sequential':
                    workloads[wl_type] = self.trace_generator.sequential_scan_trace(5000)
                elif wl_type == 'random':
                    workloads[wl_type] = self.trace_generator.random_access_trace(5000, 5000)
                elif wl_type == 'mixed':
                    workloads[wl_type] = self.trace_generator.mixed_workload_trace(5000)
        
        self.workloads = workloads
        return workloads
    
    def evaluate_baselines(self, baselines: Dict, workloads: Dict) -> Dict:
        """
        Evaluate all baseline configurations on all workloads.
        
        This establishes the performance of "one-size-fits-all" designs
        that don't adapt to specific workload characteristics.
        
        Parameters:
        -----------
        baselines : dict
            Baseline configurations
        workloads : dict
            Workload traces
            
        Returns:
        --------
        results : dict
            Nested dict: results[baseline_name][workload_name] = miss_rate
        """
        results = {}
        
        print("="*70)
        print("EVALUATING BASELINE CONFIGURATIONS")
        print("="*70)
        
        for baseline_name, config in baselines.items():
            results[baseline_name] = {}
            print(f"\nBaseline: {baseline_name}")
            print(f"   {config['description']}")
            print(f"   Size={config['cache_size']}B, Block={config['block_size']}B, "
                  f"Assoc={config['associativity']}")
            print()
            
            for workload_name, trace in workloads.items():
                miss_rate = evaluate_cache_config(
                    config['cache_size'],
                    config['block_size'],
                    config['associativity'],
                    trace
                )
                results[baseline_name][workload_name] = miss_rate
                print(f"   {workload_name:15s}: Miss Rate = {miss_rate:.4f}")
        
        print("\n" + "="*70)
        self.baseline_results = results
        return results
    
    def run_ai_optimization(self, workload_name: str, trace: List[int],
                           n_calls: int = 50, verbose: bool = False) -> Dict:
        """
        Run AI-driven optimization for a specific workload.
        
        This is the key differentiator: instead of using a fixed cache
        design, we let the AI agent learn the optimal configuration
        for each specific workload.
        
        Parameters:
        -----------
        workload_name : str
            Name of workload being optimized
        trace : List[int]
            Memory access trace
        n_calls : int
            Optimization budget (number of configurations to try)
        verbose : bool
            Print detailed optimization progress
            
        Returns:
        --------
        result : dict
            Optimization results
        """
        print(f"\nOptimizing for: {workload_name}")
        print("-" * 70)
        
        # Create objective function for this workload
        def objective(cache_size, block_size, associativity):
            return evaluate_cache_config(cache_size, block_size, 
                                        associativity, trace)
        
        # Run Bayesian optimization
        optimizer = BayesianCacheOptimizer(
            max_cache_size=self.max_cache_size,
            n_calls=n_calls,
            verbose=verbose
        )
        
        start_time = time.time()
        result = optimizer.optimize(objective)
        elapsed_time = time.time() - start_time
        
        result['workload_name'] = workload_name
        result['optimization_time'] = elapsed_time
        result['pareto_frontier'] = optimizer.get_pareto_frontier()
        
        return result
    
    def optimize_all_workloads(self, workloads: Dict, 
                              n_calls: int = 50,
                              verbose: bool = False) -> Dict:
        """
        Run AI optimization for all workloads.
        
        Parameters:
        -----------
        workloads : dict
            Dictionary of workload traces
        n_calls : int
            Optimization budget per workload
        verbose : bool
            Detailed progress output
            
        Returns:
        --------
        results : dict
            Optimization results for each workload
        """
        results = {}
        
        print("\n" + "="*70)
        print("AI-DRIVEN OPTIMIZATION FOR ALL WORKLOADS")
        print("="*70)
        
        for workload_name, trace in workloads.items():
            result = self.run_ai_optimization(
                workload_name, trace, n_calls, verbose
            )
            results[workload_name] = result
        
        self.optimized_results = results
        return results
    
    def compare_performance(self) -> Dict:
        """
        Compare AI-optimized vs. baseline performance.
        
        Key Metrics:
        1. Absolute improvement (percentage points)
        2. Relative improvement (percentage)
        3. Best baseline vs. AI-optimized
        
        Returns:
        --------
        comparison : dict
            Performance comparison statistics
        """
        if not self.baseline_results or not self.optimized_results:
            print("WARNING: Run experiments first!")
            return {}
        
        comparison = {}
        
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON: AI-OPTIMIZED vs. BASELINES")
        print("="*70)
        
        for workload_name in self.workloads.keys():
            # Get AI-optimized result
            ai_miss_rate = self.optimized_results[workload_name]['best_miss_rate']
            
            # Get baseline results
            baseline_miss_rates = {
                name: results[workload_name]
                for name, results in self.baseline_results.items()
            }
            
            # Find best baseline
            best_baseline_name = min(baseline_miss_rates, 
                                    key=baseline_miss_rates.get)
            best_baseline_miss_rate = baseline_miss_rates[best_baseline_name]
            
            # Calculate improvement
            absolute_improvement = best_baseline_miss_rate - ai_miss_rate
            if best_baseline_miss_rate > 0:
                relative_improvement = (absolute_improvement / 
                                      best_baseline_miss_rate) * 100
            else:
                relative_improvement = 0
            
            comparison[workload_name] = {
                'ai_miss_rate': ai_miss_rate,
                'best_baseline_name': best_baseline_name,
                'best_baseline_miss_rate': best_baseline_miss_rate,
                'absolute_improvement': absolute_improvement,
                'relative_improvement_pct': relative_improvement,
                'all_baselines': baseline_miss_rates
            }
            
            print(f"\n{workload_name}:")
            print(f"   AI-Optimized:     {ai_miss_rate:.4f}")
            print(f"   Best Baseline:    {best_baseline_miss_rate:.4f} ({best_baseline_name})")
            print(f"   Improvement:      {absolute_improvement:.4f} "
                  f"({relative_improvement:+.1f}%)")
            
            if relative_improvement > 0:
                print(f"   AI wins by {relative_improvement:.1f}%")
            else:
                print(f"   WARNING: Baseline competitive")
        
        print("\n" + "="*70)
        
        # Summary statistics
        improvements = [c['relative_improvement_pct'] for c in comparison.values()]
        print(f"\nSUMMARY STATISTICS:")
        print(f"   Average Improvement: {np.mean(improvements):.1f}%")
        print(f"   Median Improvement:  {np.median(improvements):.1f}%")
        print(f"   Best Improvement:    {np.max(improvements):.1f}%")
        print(f"   Worst Case:          {np.min(improvements):.1f}%")
        print("="*70)
        
        return comparison
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj
    
    def save_results(self, filename: str = 'experiment_results.json'):
        """
        Save all experimental results to JSON file.
        
        Parameters:
        -----------
        filename : str
            Output filename
        """
        results = {
            'baselines': self.baseline_results,
            'optimized': {
                name: {
                    'best_config': result['best_config'],
                    'best_miss_rate': result['best_miss_rate'],
                    'optimization_time': result['optimization_time'],
                    'pareto_frontier': result['pareto_frontier']
                }
                for name, result in self.optimized_results.items()
            },
            'workload_stats': {
                name: {
                    'num_accesses': len(trace),
                    'unique_addresses': len(set(trace))
                }
                for name, trace in self.workloads.items()
            }
        }
        
        # Convert numpy types to native Python types
        results = self._convert_numpy_types(results)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def run_full_experiment(self, n_calls: int = 50, 
                           save_results: bool = True) -> Dict:
        """
        Run complete experimental workflow.
        
        Workflow:
        1. Generate workloads
        2. Define baselines
        3. Evaluate baselines
        4. Run AI optimization
        5. Compare performance
        6. Save results
        
        Parameters:
        -----------
        n_calls : int
            Optimization budget per workload
        save_results : bool
            Save results to JSON
            
        Returns:
        --------
        results : dict
            Complete experimental results
        """
        print("="*70)
        print("SMARTCACHE EXPERIMENT - FULL WORKFLOW")
        print("="*70)
        print(f"Max Cache Size: {self.max_cache_size} bytes "
              f"({self.max_cache_size // 1024} KB)")
        print(f"Optimization Budget: {n_calls} evaluations per workload")
        print("="*70)
        
        # Step 1: Generate workloads
        print("\nGenerating workloads...")
        workloads = self.generate_workloads()
        print(f"   Generated {len(workloads)} workload traces")
        
        # Step 2: Define baselines
        print("\nDefining baseline configurations...")
        baselines = self.define_baselines()
        print(f"   Defined {len(baselines)} baseline configurations")
        
        # Step 3: Evaluate baselines
        baseline_results = self.evaluate_baselines(baselines, workloads)
        
        # Step 4: AI optimization
        optimized_results = self.optimize_all_workloads(
            workloads, n_calls=n_calls, verbose=False
        )
        
        # Step 5: Performance comparison
        comparison = self.compare_performance()
        
        # Step 6: Save results
        if save_results:
            self.save_results()
        
        return {
            'baselines': baseline_results,
            'optimized': optimized_results,
            'comparison': comparison
        }


if __name__ == "__main__":
    # Run demonstration experiment
    print("SmartCache Experiment Framework - Demonstration")
    print()
    
    # Create experiment with smaller budget for demo
    experiment = SmartCacheExperiment(max_cache_size=32768, seed=42)
    
    # Run with reduced workload for quick demonstration
    print("Running quick demonstration (reduced workload)...\n")
    
    # Generate smaller workload subset
    workloads = {
        'matmul_32': experiment.trace_generator.matrix_multiplication_trace(32),
        'sort_1k': experiment.trace_generator.quicksort_trace(1000),
        'sequential': experiment.trace_generator.sequential_scan_trace(2000),
    }
    
    # Run experiment
    baselines = experiment.define_baselines()
    experiment.evaluate_baselines(baselines, workloads)
    experiment.optimize_all_workloads(workloads, n_calls=20, verbose=False)
    experiment.compare_performance()
    
    print("\nDemonstration complete!")
    print("Run with full parameters for comprehensive results.")
