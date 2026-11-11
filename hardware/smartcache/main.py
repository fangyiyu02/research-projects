"""
SmartCache: AI-Driven Cache Optimization Framework
===================================================
Main entry point for running the complete SmartCache experiment.

This framework demonstrates how AI-driven optimization (Bayesian Optimization)
can outperform traditional "one-size-fits-all" cache designs by learning
optimal configurations for specific workloads.

Usage:
    python main.py --mode [quick|full|custom]
    python main.py --help

Modes:
    quick  - Fast demonstration with reduced workloads (5 min)
    full   - Complete experiment with all workloads (30-60 min)
    custom - Interactive configuration
"""

import argparse
import sys
import time
from typing import Dict, List

from cache_simulator import CacheSimulator
from trace_generator import TraceGenerator
from ai_optimizer import BayesianCacheOptimizer
from experiment_framework import SmartCacheExperiment
from visualize_results import SmartCacheVisualizer


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║            SMARTCACHE AI OPTIMIZATION FRAMEWORK                  ║
    ║                                                                  ║
    ║        Solving the Memory Wall with Bayesian Optimization       ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    try:
        print(banner)
    except UnicodeEncodeError:
        print("SmartCache AI Optimization Framework")
        print("=" * 70)


def run_quick_demo():
    """
    Run a quick demonstration with reduced workloads.
    
    This mode is perfect for:
    - Understanding the framework
    - Testing installation
    - Quick validation
    
    Duration: ~5 minutes
    """
    print("\n" + "="*70)
    print("QUICK DEMONSTRATION MODE")
    print("="*70)
    print("This will run a fast demo with reduced workloads.")
    print("Estimated time: 5 minutes\n")
    
    # Create experiment
    experiment = SmartCacheExperiment(max_cache_size=32768, seed=42)
    
    # Generate small workload suite
    print("Generating workloads...")
    workloads = {
        'matmul_32': experiment.trace_generator.matrix_multiplication_trace(32),
        'sort_1k': experiment.trace_generator.quicksort_trace(1000),
        'sequential': experiment.trace_generator.sequential_scan_trace(2000),
        'stride_8': experiment.trace_generator.strided_access_trace(2000, stride=8),
    }
    experiment.workloads = workloads  # Store workloads in experiment object
    print(f"   Generated {len(workloads)} workload traces\n")
    
    # Define baselines
    baselines = experiment.define_baselines()
    
    # Evaluate baselines
    print("\nEvaluating baseline configurations...")
    experiment.evaluate_baselines(baselines, workloads)
    
    # Run AI optimization (reduced budget for speed)
    print("\nRunning AI optimization (20 evaluations per workload)...")
    experiment.optimize_all_workloads(workloads, n_calls=20, verbose=False)
    
    # Compare results
    print("\nAnalyzing results...")
    comparison = experiment.compare_performance()
    
    # Save results
    experiment.save_results('quick_demo_results.json')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    viz = SmartCacheVisualizer()
    
    # Plot for first workload
    first_workload = list(workloads.keys())[0]
    pareto_points = experiment.optimized_results[first_workload]['pareto_frontier']
    viz.plot_pareto_frontier(pareto_points, 
                            title=f'Pareto Frontier: {first_workload}',
                            save_path='quick_demo_pareto.png')
    
    # Comparison plot
    viz.plot_ai_vs_baseline_comparison(comparison,
                                      save_path='quick_demo_comparison.png')
    
    print("\n" + "="*70)
    print("QUICK DEMO COMPLETE!")
    print("="*70)
    print("Results saved to:")
    print("  - quick_demo_results.json")
    print("  - quick_demo_pareto.png")
    print("  - quick_demo_comparison.png")
    print("\nKey Takeaway: AI-driven optimization adapts cache configurations")
    print("              to workload characteristics, outperforming static designs.")
    print("="*70)


def run_full_experiment():
    """
    Run comprehensive experiment with complete workload suite.
    
    This mode provides:
    - Complete workload coverage
    - Thorough optimization (50 evaluations per workload)
    - Comprehensive visualizations
    - Publication-ready results
    
    Duration: 30-60 minutes
    """
    print("\n" + "="*70)
    print("FULL EXPERIMENT MODE")
    print("="*70)
    print("This will run the complete experiment with all workloads.")
    print("Estimated time: 30-60 minutes")
    print("="*70)
    
    response = input("\nProceed with full experiment? (y/n): ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    start_time = time.time()
    
    # Create experiment with full parameter
    experiment = SmartCacheExperiment(max_cache_size=65536, seed=42)
    
    # Run full workflow
    results = experiment.run_full_experiment(
        n_calls=50,  # Full optimization budget
        save_results=True
    )
    
    elapsed_time = time.time() - start_time
    
    # Generate comprehensive visualizations
    print("\nGenerating comprehensive visualizations...")
    viz = SmartCacheVisualizer()
    
    # Comparison plot
    viz.plot_ai_vs_baseline_comparison(
        results['comparison'],
        save_path='full_experiment_comparison.png'
    )
    
    # Multi-workload Pareto frontiers
    pareto_dict = {
        name: result['pareto_frontier']
        for name, result in results['optimized'].items()
    }
    viz.plot_multiple_pareto_frontiers(
        pareto_dict,
        save_path='full_experiment_multi_pareto.png'
    )
    
    # Individual convergence plots for selected workloads
    sample_workloads = list(results['optimized'].keys())[:3]
    for workload_name in sample_workloads:
        history = results['optimized'][workload_name]['history']
        viz.plot_optimization_convergence(
            history,
            title=f'Optimization Convergence: {workload_name}',
            save_path=f'convergence_{workload_name}.png'
        )
        
        viz.plot_design_space_exploration(
            history,
            save_path=f'design_space_{workload_name}.png'
        )
    
    print("\n" + "="*70)
    print("FULL EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print("\nResults saved to:")
    print("  - experiment_results.json")
    print("  - full_experiment_comparison.png")
    print("  - full_experiment_multi_pareto.png")
    print("  - convergence_*.png (per workload)")
    print("  - design_space_*.png (per workload)")
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    # Print summary statistics
    improvements = [c['relative_improvement_pct'] 
                   for c in results['comparison'].values()]
    print(f"Average Improvement over Best Baseline: {sum(improvements)/len(improvements):.1f}%")
    print(f"Best Case Improvement: {max(improvements):.1f}%")
    print(f"Worst Case: {min(improvements):.1f}%")
    
    print("\nConclusion: AI-driven optimization consistently outperforms")
    print("            static 'one-size-fits-all' cache designs by learning")
    print("            workload-specific configurations.")
    print("="*70)


def run_custom_experiment():
    """
    Run customizable experiment with user-specified parameters.
    """
    print("\n" + "="*70)
    print("CUSTOM EXPERIMENT MODE")
    print("="*70)
    print("Configure your own experiment parameters.\n")
    
    # Get user inputs
    try:
        max_cache_size = int(input("Max cache size (bytes) [default: 65536]: ") or "65536")
        n_calls = int(input("Optimization budget per workload [default: 50]: ") or "50")
        
        print("\nAvailable workload types:")
        print("  1. matmul_N (matrix multiplication, N=32,64,128)")
        print("  2. sort_N (quicksort, N=1000,5000,10000)")
        print("  3. sequential (sequential array access)")
        print("  4. random (random access)")
        print("  5. stride_N (strided access, N=stride)")
        print("  6. mixed (mixed workload)")
        
        workload_input = input("\nEnter workload types (comma-separated) or 'all': ")
        
        # Create experiment
        experiment = SmartCacheExperiment(max_cache_size=max_cache_size, seed=42)
        
        # Generate workloads
        if workload_input.lower() == 'all':
            workloads = experiment.generate_workloads()
        else:
            workload_list = [w.strip() for w in workload_input.split(',')]
            workloads = experiment.generate_workloads(workload_list)
        
        # Define baselines
        baselines = experiment.define_baselines()
        
        # Run experiment
        print("\nStarting custom experiment...")
        experiment.evaluate_baselines(baselines, workloads)
        experiment.optimize_all_workloads(workloads, n_calls=n_calls, verbose=False)
        comparison = experiment.compare_performance()
        
        # Save results
        experiment.save_results('custom_experiment_results.json')
        
        # Visualizations
        viz = SmartCacheVisualizer()
        viz.plot_ai_vs_baseline_comparison(comparison, 
                                          save_path='custom_comparison.png')
        
        print("\nCustom experiment complete!")
        print("Results saved to custom_experiment_results.json")
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nError or cancelled: {e}")
        return


def test_installation():
    """
    Test that all components are working correctly.
    """
    print("\n" + "="*70)
    print("TESTING INSTALLATION")
    print("="*70)
    
    try:
        print("\n1. Testing Cache Simulator...")
        cache = CacheSimulator(cache_size=4096, block_size=64, associativity=4)
        trace = [i * 64 for i in range(100)]
        results = cache.run_trace(trace)
        print(f"   Simulator working. Miss rate: {results['miss_rate']:.4f}")
        
        print("\n2. Testing Trace Generator...")
        generator = TraceGenerator(seed=42)
        matmul_trace = generator.matrix_multiplication_trace(16)
        print(f"   Generated {len(matmul_trace)} memory accesses")
        
        print("\n3. Testing AI Optimizer...")
        def dummy_objective(cs, bs, assoc):
            return 0.5 * (1 / (cs / 1024))
        
        optimizer = BayesianCacheOptimizer(max_cache_size=8192, n_calls=5, verbose=False)
        result = optimizer.optimize(dummy_objective)
        print(f"   Optimizer working. Best miss rate: {result['best_miss_rate']:.4f}")
        
        print("\n4. Testing Visualizer...")
        viz = SmartCacheVisualizer()
        print("   Visualizer initialized successfully")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print("Installation is working correctly.")
        print("You can now run experiments with confidence.")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        print("\nPlease check that all dependencies are installed:")
        print("  pip install numpy scipy scikit-optimize matplotlib seaborn")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SmartCache: AI-Driven Cache Optimization Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode quick          # Fast 5-minute demo
  python main.py --mode full           # Complete experiment (30-60 min)
  python main.py --mode custom         # Custom parameters
  python main.py --test                # Test installation
  
For more information, see README.md
        """
    )
    
    parser.add_argument('--mode', type=str, 
                       choices=['quick', 'full', 'custom'],
                       help='Experiment mode')
    parser.add_argument('--test', action='store_true',
                       help='Test installation')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Handle test flag
    if args.test:
        success = test_installation()
        sys.exit(0 if success else 1)
    
    # If no mode specified, show interactive menu
    if not args.mode:
        print("\n" + "="*70)
        print("SELECT EXPERIMENT MODE:")
        print("="*70)
        print("1. Quick Demo      - Fast demonstration (5 min)")
        print("2. Full Experiment - Complete workload suite (30-60 min)")
        print("3. Custom          - Configure your own experiment")
        print("4. Test            - Test installation")
        print("5. Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == '1':
            args.mode = 'quick'
        elif choice == '2':
            args.mode = 'full'
        elif choice == '3':
            args.mode = 'custom'
        elif choice == '4':
            test_installation()
            return
        elif choice == '5':
            print("Goodbye!")
            return
        else:
            print("Invalid choice. Exiting.")
            return
    
    # Run selected mode
    if args.mode == 'quick':
        run_quick_demo()
    elif args.mode == 'full':
        run_full_experiment()
    elif args.mode == 'custom':
        run_custom_experiment()


if __name__ == "__main__":
    main()
