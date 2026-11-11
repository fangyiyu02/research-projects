"""
AI Architect Agent - Bayesian Optimization for Cache Design
============================================================
Implements intelligent exploration of the cache design space using
Bayesian Optimization to find optimal configurations.

Key Concepts:
- Surrogate Model: Gaussian Process that learns the objective function
- Acquisition Function: Guides exploration vs. exploitation trade-off
- Sequential Decision Making: Iteratively refines search based on feedback

This approach dramatically outperforms brute-force grid search by learning
from previous evaluations and focusing on promising regions.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')


class BayesianCacheOptimizer:
    """
    AI-driven cache configuration optimizer using Bayesian Optimization.
    
    Bayesian Optimization is ideal for expensive black-box optimization:
    1. Build a probabilistic surrogate model (Gaussian Process) of the objective
    2. Use acquisition function to decide where to sample next
    3. Update model with new observation
    4. Repeat until convergence or budget exhausted
    
    The acquisition function balances:
    - EXPLOITATION: Sample where model predicts good performance
    - EXPLORATION: Sample where model is uncertain
    
    Common acquisition functions:
    - Expected Improvement (EI): Expected improvement over current best
    - Lower Confidence Bound (LCB): Mean - kappa * std_dev
    - Probability of Improvement (PI): Probability of beating current best
    """
    
    def __init__(self, 
                 max_cache_size: int = 65536,  # 64 KB max
                 min_cache_size: int = 1024,   # 1 KB min
                 n_calls: int = 50,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize the Bayesian optimizer.
        
        Parameters:
        -----------
        max_cache_size : int
            Maximum allowed cache size in bytes (constraint)
        min_cache_size : int
            Minimum cache size in bytes
        n_calls : int
            Number of optimization iterations (function evaluations)
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Print optimization progress
        """
        self.max_cache_size = max_cache_size
        self.min_cache_size = min_cache_size
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        
        # Optimization history
        self.history = []
        self.best_config = None
        self.best_miss_rate = float('inf')
        
        # Define search space (design space knobs)
        # These are the parameters the AI will tune
        self._define_search_space()
    
    def _define_search_space(self):
        """
        Define the design space dimensions.
        
        Search Space Design Philosophy:
        - Cache size: Logarithmic scale for efficiency (powers of 2 are typical)
        - Block size: Powers of 2 (hardware requirement)
        - Associativity: Powers of 2 (common in practice)
        
        The search space size is HUGE:
        - Cache size options: ~6 choices (1KB to 64KB)
        - Block size options: ~6 choices (16B to 512B)
        - Associativity options: ~5 choices (1 to 16-way)
        - Total combinations: 6 * 6 * 5 = 180+
        
        But with non-linear interactions, effective space is much larger!
        Bayesian Optimization explores this intelligently.
        """
        # Cache size: 1KB to 64KB (powers of 2)
        # Represented as exponent: 2^10 to 2^16
        self.space = [
            Integer(10, 16, name='cache_size_exp'),  # 1KB to 64KB
            Integer(4, 9, name='block_size_exp'),     # 16B to 512B
            Integer(0, 4, name='assoc_exp')           # 1-way to 16-way (2^0 to 2^4)
        ]
        
        self.dimension_names = ['cache_size_exp', 'block_size_exp', 'assoc_exp']
    
    def _decode_params(self, cache_size_exp: int, block_size_exp: int, 
                       assoc_exp: int) -> Tuple[int, int, int]:
        """
        Decode exponential parameters to actual values.
        
        Why exponential encoding?
        - Cache parameters are typically powers of 2 in hardware
        - Provides more uniform sampling in log space
        - Matches how designers think about cache sizes
        
        Parameters:
        -----------
        cache_size_exp : int
            Exponent for cache size (2^exp bytes)
        block_size_exp : int
            Exponent for block size (2^exp bytes)
        assoc_exp : int
            Exponent for associativity (2^exp way)
            
        Returns:
        --------
        cache_size, block_size, associativity : Tuple[int, int, int]
            Decoded parameter values
        """
        cache_size = 2 ** cache_size_exp
        block_size = 2 ** block_size_exp
        associativity = 2 ** assoc_exp
        
        # Constraint: cache size must not exceed maximum
        if cache_size > self.max_cache_size:
            cache_size = self.max_cache_size
        
        # Constraint: ensure valid configuration
        # (cache must have at least one set)
        num_blocks = cache_size // block_size
        if associativity > num_blocks:
            associativity = num_blocks
        
        return cache_size, block_size, associativity
    
    def _validate_config(self, cache_size: int, block_size: int, 
                        associativity: int) -> bool:
        """
        Validate cache configuration against constraints.
        
        Design Constraints:
        1. Cache size must not exceed maximum (area/power budget)
        2. Block size must divide cache size evenly
        3. Associativity must be achievable with given cache geometry
        4. Must have at least one cache set
        
        Parameters:
        -----------
        cache_size, block_size, associativity : int
            Cache parameters to validate
            
        Returns:
        --------
        valid : bool
            True if configuration is valid
        """
        if cache_size > self.max_cache_size or cache_size < self.min_cache_size:
            return False
        
        if block_size <= 0 or cache_size % block_size != 0:
            return False
        
        num_blocks = cache_size // block_size
        if associativity <= 0 or associativity > num_blocks:
            return False
        
        num_sets = num_blocks // associativity
        if num_sets <= 0:
            return False
        
        return True
    
    def optimize(self, objective_function: Callable, 
                 acq_func: str = 'EI') -> Dict:
        """
        Run Bayesian Optimization to find optimal cache configuration.
        
        Bayesian Optimization Algorithm:
        --------------------------------
        1. Initialize: Sample a few random configurations (exploration)
        2. Build Surrogate Model: Fit Gaussian Process to observed data
           - GP models objective function f(x) as a distribution
           - Provides mean prediction and uncertainty estimate
        3. Acquisition Function: Use GP to decide next point to sample
           - EI (Expected Improvement): E[max(f_best - f(x), 0)]
           - Balances exploring uncertain regions vs. exploiting good regions
        4. Evaluate: Run cache simulator at selected configuration
        5. Update: Add result to dataset, update GP
        6. Repeat steps 2-5 until budget exhausted
        
        Why this works:
        - GP learns correlations between parameters and performance
        - Discovers non-linear relationships (e.g., block size vs. miss rate)
        - Converges faster than grid search by avoiding bad regions
        - Provides uncertainty quantification
        
        Parameters:
        -----------
        objective_function : Callable
            Function that takes (cache_size, block_size, associativity) 
            and returns miss_rate
        acq_func : str
            Acquisition function: 'EI', 'LCB', or 'PI'
            
        Returns:
        --------
        result : dict
            Optimization results including best configuration and history
        """
        self.history = []
        self.best_config = None
        self.best_miss_rate = float('inf')
        
        # Wrapper to interface with skopt
        @use_named_args(self.space)
        def objective_wrapper(**params):
            """
            Wrapper function for the optimizer.
            
            This function:
            1. Decodes parameters from search space
            2. Validates configuration
            3. Calls simulator (expensive operation)
            4. Records result
            5. Returns miss rate (to be minimized)
            """
            cache_size, block_size, associativity = self._decode_params(
                params['cache_size_exp'],
                params['block_size_exp'],
                params['assoc_exp']
            )
            
            # Validate configuration
            if not self._validate_config(cache_size, block_size, associativity):
                # Return high penalty for invalid configurations
                return 1.0
            
            # Evaluate objective (run cache simulator)
            try:
                miss_rate = objective_function(cache_size, block_size, associativity)
            except Exception as e:
                if self.verbose:
                    print(f"Error evaluating config: {e}")
                return 1.0
            
            # Record in history
            config = {
                'cache_size': cache_size,
                'block_size': block_size,
                'associativity': associativity,
                'miss_rate': miss_rate
            }
            self.history.append(config)
            
            # Update best found
            if miss_rate < self.best_miss_rate:
                self.best_miss_rate = miss_rate
                self.best_config = config.copy()
                if self.verbose:
                    print(f"\n🎯 New best: Miss Rate = {miss_rate:.4f}")
                    print(f"   Config: Size={cache_size}B, Block={block_size}B, "
                          f"Assoc={associativity}-way")
            
            return miss_rate
        
        # Run Bayesian Optimization
        # The optimizer uses Gaussian Process regression as surrogate model
        if self.verbose:
            print("="*70)
            print("🤖 AI ARCHITECT AGENT - BAYESIAN OPTIMIZATION")
            print("="*70)
            print(f"Search Space: {len(self.space)} dimensions")
            print(f"Max Cache Size: {self.max_cache_size} bytes "
                  f"({self.max_cache_size // 1024} KB)")
            print(f"Optimization Budget: {self.n_calls} evaluations")
            print(f"Acquisition Function: {acq_func}")
            print("="*70 + "\n")
        
        # gp_minimize performs Gaussian Process-based Bayesian Optimization
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=self.space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            acq_func=acq_func,  # Expected Improvement
            n_initial_points=10,  # Random exploration first
            verbose=self.verbose
        )
        
        if self.verbose:
            print("\n" + "="*70)
            print("✅ OPTIMIZATION COMPLETE")
            print("="*70)
            print(f"Best Miss Rate: {self.best_miss_rate:.4f}")
            print(f"Best Configuration:")
            print(f"  Cache Size: {self.best_config['cache_size']} bytes "
                  f"({self.best_config['cache_size'] // 1024} KB)")
            print(f"  Block Size: {self.best_config['block_size']} bytes")
            print(f"  Associativity: {self.best_config['associativity']}-way")
            print("="*70 + "\n")
        
        return {
            'best_config': self.best_config,
            'best_miss_rate': self.best_miss_rate,
            'history': self.history,
            'skopt_result': result
        }
    
    def get_pareto_frontier(self) -> List[Dict]:
        """
        Extract Pareto frontier from optimization history.
        
        Pareto Frontier:
        - Set of configurations where no other configuration is better in all objectives
        - Trade-off curve between cache size (cost) and miss rate (performance)
        - A point is Pareto-optimal if improving one objective worsens another
        
        For cache design:
        - Objective 1: Minimize miss rate (performance)
        - Objective 2: Minimize cache size (cost/area/power)
        
        Returns:
        --------
        pareto_points : List[Dict]
            List of Pareto-optimal configurations sorted by cache size
        """
        if not self.history:
            return []
        
        pareto_points = []
        
        # Sort by cache size for efficient Pareto extraction
        sorted_history = sorted(self.history, key=lambda x: x['cache_size'])
        
        min_miss_rate_so_far = float('inf')
        
        for config in sorted_history:
            # A point is Pareto-optimal if it has the best miss rate
            # seen so far for its cache size or smaller
            if config['miss_rate'] < min_miss_rate_so_far:
                pareto_points.append(config)
                min_miss_rate_so_far = config['miss_rate']
        
        return pareto_points


if __name__ == "__main__":
    # Demonstration with a dummy objective function
    print("AI Optimizer - Standalone Test")
    print("="*60)
    
    # Dummy objective function (in practice, this would be cache simulation)
    def dummy_objective(cache_size, block_size, associativity):
        """Synthetic objective with non-linear behavior."""
        # Favor larger caches and higher associativity
        # Add some non-linear terms to make it interesting
        miss_rate = 0.5 * (1 / (cache_size / 1024)) + \
                   0.3 * (1 / associativity) + \
                   0.2 * (block_size / 64) ** 2
        # Add noise
        miss_rate += np.random.normal(0, 0.01)
        return max(0.0, min(1.0, miss_rate))
    
    # Run optimization
    optimizer = BayesianCacheOptimizer(
        max_cache_size=32768,  # 32 KB
        n_calls=30,
        verbose=True
    )
    
    result = optimizer.optimize(dummy_objective)
    
    print("\nPareto Frontier:")
    pareto = optimizer.get_pareto_frontier()
    for i, point in enumerate(pareto):
        print(f"{i+1}. Size={point['cache_size']:5d}B, "
              f"Block={point['block_size']:3d}B, "
              f"Assoc={point['associativity']:2d}, "
              f"Miss={point['miss_rate']:.4f}")
