"""
SmartCache Simulator - Lightweight Trace-Driven Cache Simulator
================================================================
Implements a configurable cache with LRU replacement policy.
Designed to be modular and interface-ready for Verilog implementation.

Key Features:
- Configurable cache size, block size, and associativity
- LRU (Least Recently Used) replacement policy
- Miss rate calculation as primary performance metric
"""

import math
from typing import List, Tuple, Dict
from collections import OrderedDict


class CacheSimulator:
    """
    Trace-driven cache simulator with configurable parameters.
    
    Parameters:
    -----------
    cache_size : int
        Total cache size in bytes
    block_size : int
        Cache block/line size in bytes (must be power of 2)
    associativity : int
        Set associativity (1=direct-mapped, N=N-way, cache_size=fully-associative)
    
    Architecture:
    ------------
    - Address breakdown: [Tag | Index | Offset]
    - Offset bits: log2(block_size)
    - Index bits: log2(num_sets)
    - Tag bits: Remaining address bits
    """
    
    def __init__(self, cache_size: int, block_size: int, associativity: int):
        # Validate parameters
        assert cache_size > 0, "Cache size must be positive"
        assert block_size > 0 and (block_size & (block_size - 1)) == 0, \
            "Block size must be power of 2"
        assert associativity > 0, "Associativity must be positive"
        assert cache_size >= block_size * associativity, \
            "Cache size must accommodate at least one set"
        
        self.cache_size = cache_size
        self.block_size = block_size
        self.associativity = associativity
        
        # Calculate cache geometry
        self.num_blocks = cache_size // block_size
        self.num_sets = self.num_blocks // associativity
        
        # Address bit fields
        self.offset_bits = int(math.log2(block_size))
        self.index_bits = int(math.log2(self.num_sets))
        
        # Initialize cache structure: list of sets, each set is an OrderedDict for LRU
        # OrderedDict maintains insertion order, we'll move accessed items to end
        self.cache = [OrderedDict() for _ in range(self.num_sets)]
        
        # Performance counters
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0
        
    def _parse_address(self, address: int) -> Tuple[int, int, int]:
        """
        Parse memory address into tag, index, and offset.
        
        Returns:
        --------
        tag : int
            Tag bits for comparison
        index : int
            Set index
        offset : int
            Block offset (not used in cache logic but included for completeness)
        """
        offset = address & ((1 << self.offset_bits) - 1)
        index = (address >> self.offset_bits) & ((1 << self.index_bits) - 1)
        tag = address >> (self.offset_bits + self.index_bits)
        return tag, index, offset
    
    def access(self, address: int) -> bool:
        """
        Simulate a cache access (read or write - unified cache).
        
        Parameters:
        -----------
        address : int
            Memory address to access
            
        Returns:
        --------
        hit : bool
            True if cache hit, False if cache miss
        """
        self.total_accesses += 1
        tag, index, offset = self._parse_address(address)
        
        cache_set = self.cache[index]
        
        # Check for cache hit
        if tag in cache_set:
            self.hits += 1
            # LRU: Move accessed block to end (most recently used)
            cache_set.move_to_end(tag)
            return True
        
        # Cache miss
        self.misses += 1
        
        # Check if set is full
        if len(cache_set) >= self.associativity:
            # Evict LRU block (first item in OrderedDict)
            cache_set.popitem(last=False)
        
        # Insert new block (at end = most recently used)
        cache_set[tag] = True
        
        return False
    
    def run_trace(self, trace: List[int]) -> Dict[str, float]:
        """
        Run a complete memory access trace through the simulator.
        
        Parameters:
        -----------
        trace : List[int]
            List of memory addresses to access
            
        Returns:
        --------
        results : dict
            Performance metrics including miss rate
        """
        for address in trace:
            self.access(address)
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate and return performance statistics.
        
        Returns:
        --------
        stats : dict
            Dictionary containing:
            - miss_rate: Fraction of accesses that missed
            - hit_rate: Fraction of accesses that hit
            - total_accesses: Total number of memory accesses
            - hits: Number of cache hits
            - misses: Number of cache misses
        """
        if self.total_accesses == 0:
            return {
                'miss_rate': 0.0,
                'hit_rate': 0.0,
                'total_accesses': 0,
                'hits': 0,
                'misses': 0
            }
        
        miss_rate = self.misses / self.total_accesses
        hit_rate = self.hits / self.total_accesses
        
        return {
            'miss_rate': miss_rate,
            'hit_rate': hit_rate,
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'misses': self.misses
        }
    
    def reset(self):
        """Reset cache state and performance counters."""
        self.cache = [OrderedDict() for _ in range(self.num_sets)]
        self.hits = 0
        self.misses = 0
        self.total_accesses = 0
    
    def get_config(self) -> Dict[str, int]:
        """
        Get current cache configuration.
        
        Returns:
        --------
        config : dict
            Cache parameters and derived geometry
        """
        return {
            'cache_size': self.cache_size,
            'block_size': self.block_size,
            'associativity': self.associativity,
            'num_blocks': self.num_blocks,
            'num_sets': self.num_sets,
            'offset_bits': self.offset_bits,
            'index_bits': self.index_bits
        }
    
    def __str__(self) -> str:
        """String representation of cache configuration."""
        config = self.get_config()
        return (f"Cache Configuration:\n"
                f"  Size: {config['cache_size']} B ({config['cache_size']//1024} KB)\n"
                f"  Block Size: {config['block_size']} B\n"
                f"  Associativity: {config['associativity']}-way\n"
                f"  Number of Sets: {config['num_sets']}\n"
                f"  Number of Blocks: {config['num_blocks']}")


def evaluate_cache_config(cache_size: int, block_size: int, 
                         associativity: int, trace: List[int]) -> float:
    """
    Convenience function to evaluate a cache configuration on a trace.
    
    This function provides a simple interface for the optimization agent.
    
    Parameters:
    -----------
    cache_size : int
        Total cache size in bytes
    block_size : int
        Block size in bytes
    associativity : int
        Set associativity
    trace : List[int]
        Memory access trace
        
    Returns:
    --------
    miss_rate : float
        Miss rate for this configuration [0.0, 1.0]
    """
    try:
        simulator = CacheSimulator(cache_size, block_size, associativity)
        results = simulator.run_trace(trace)
        return results['miss_rate']
    except (AssertionError, ValueError) as e:
        # Invalid configuration, return worst possible miss rate
        return 1.0


if __name__ == "__main__":
    # Quick test of the simulator
    print("="*60)
    print("SmartCache Simulator - Test Run")
    print("="*60)
    
    # Create a simple test trace (sequential access pattern)
    test_trace = [i * 64 for i in range(1000)]  # 64-byte stride
    
    # Test configuration
    cache = CacheSimulator(cache_size=4096, block_size=64, associativity=4)
    print(cache)
    print()
    
    # Run simulation
    results = cache.run_trace(test_trace)
    
    print("Simulation Results:")
    print(f"  Total Accesses: {results['total_accesses']}")
    print(f"  Hits: {results['hits']}")
    print(f"  Misses: {results['misses']}")
    print(f"  Miss Rate: {results['miss_rate']:.4f}")
    print(f"  Hit Rate: {results['hit_rate']:.4f}")
    print("="*60)
