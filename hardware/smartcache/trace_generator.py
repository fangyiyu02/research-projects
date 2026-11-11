"""
Memory Access Trace Generator for Standard Algorithms
======================================================
Generates realistic memory access patterns for:
- Matrix Multiplication
- Sorting algorithms
- Array traversals
- Random access patterns

These traces simulate real application behavior for cache evaluation.
"""

import numpy as np
from typing import List, Tuple
import random


class TraceGenerator:
    """
    Generate memory access traces for various algorithms.
    
    All addresses are byte addresses aligned to 4-byte (32-bit) boundaries
    to simulate typical memory access patterns.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    @staticmethod
    def _align_address(address: int, alignment: int = 4) -> int:
        """Align address to specified byte boundary."""
        return (address // alignment) * alignment
    
    def matrix_multiplication_trace(self, matrix_size: int, 
                                    base_addr: int = 0x10000) -> List[int]:
        """
        Generate trace for Matrix Multiplication (A * B = C).
        
        Classic example of the memory wall problem:
        - Poor locality for matrix B (column-wise access)
        - Good locality for matrix A (row-wise access)
        - Good locality for matrix C (reused in inner loop)
        
        Algorithm (simplified):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    C[i][j] += A[i][k] * B[k][j]
        
        Parameters:
        -----------
        matrix_size : int
            Dimension N of NxN matrices
        base_addr : int
            Base memory address
            
        Returns:
        --------
        trace : List[int]
            Sequence of memory addresses accessed
        """
        N = matrix_size
        element_size = 4  # 4 bytes per element (float32)
        
        # Memory layout: A, B, C stored consecutively
        A_base = base_addr
        B_base = A_base + N * N * element_size
        C_base = B_base + N * N * element_size
        
        trace = []
        
        for i in range(N):
            for j in range(N):
                # Access C[i][j] for initialization (write)
                c_addr = C_base + (i * N + j) * element_size
                trace.append(c_addr)
                
                for k in range(N):
                    # Read A[i][k]
                    a_addr = A_base + (i * N + k) * element_size
                    trace.append(a_addr)
                    
                    # Read B[k][j] - Poor spatial locality!
                    b_addr = B_base + (k * N + j) * element_size
                    trace.append(b_addr)
                    
                    # Read-Modify-Write C[i][j]
                    trace.append(c_addr)
        
        return trace
    
    def quicksort_trace(self, array_size: int, 
                       base_addr: int = 0x20000) -> List[int]:
        """
        Generate trace for QuickSort algorithm.
        
        Characteristics:
        - Irregular access pattern
        - Depends on pivot selection
        - Mix of sequential and random accesses
        
        Parameters:
        -----------
        array_size : int
            Number of elements to sort
        base_addr : int
            Base memory address
            
        Returns:
        --------
        trace : List[int]
            Sequence of memory addresses accessed
        """
        element_size = 4
        trace = []
        
        # Create array with random values for realistic partitioning
        arr = list(range(array_size))
        random.shuffle(arr)
        
        def quicksort_trace_helper(low: int, high: int):
            if low < high:
                # Partition phase - many comparisons and swaps
                pivot_idx = random.randint(low, high)
                
                # Access pivot
                pivot_addr = base_addr + pivot_idx * element_size
                trace.append(pivot_addr)
                
                # Scan through array for partitioning
                for i in range(low, high + 1):
                    addr = base_addr + i * element_size
                    trace.append(addr)  # Compare with pivot
                    
                    # Simulate swaps (multiple accesses)
                    if random.random() < 0.3:  # ~30% swap rate
                        trace.append(addr)  # Write back
                
                # Recursive calls
                pivot_pos = low + (high - low) // 2
                quicksort_trace_helper(low, pivot_pos - 1)
                quicksort_trace_helper(pivot_pos + 1, high)
        
        quicksort_trace_helper(0, array_size - 1)
        return trace
    
    def sequential_scan_trace(self, array_size: int, 
                             stride: int = 1,
                             base_addr: int = 0x30000) -> List[int]:
        """
        Generate trace for sequential array access.
        
        Best-case scenario for caches:
        - Perfect spatial locality
        - Predictable access pattern
        
        Parameters:
        -----------
        array_size : int
            Number of elements
        stride : int
            Access stride (1=sequential, >1=strided access)
        base_addr : int
            Base memory address
            
        Returns:
        --------
        trace : List[int]
            Sequence of memory addresses accessed
        """
        element_size = 4
        trace = []
        
        for i in range(0, array_size, stride):
            addr = base_addr + i * element_size
            trace.append(addr)
        
        return trace
    
    def random_access_trace(self, array_size: int, 
                           num_accesses: int,
                           base_addr: int = 0x40000) -> List[int]:
        """
        Generate trace for random memory accesses.
        
        Worst-case scenario for caches:
        - No spatial locality
        - No temporal locality (if array is large)
        
        Parameters:
        -----------
        array_size : int
            Size of address space
        num_accesses : int
            Number of random accesses to generate
        base_addr : int
            Base memory address
            
        Returns:
        --------
        trace : List[int]
            Sequence of memory addresses accessed
        """
        element_size = 4
        trace = []
        
        for _ in range(num_accesses):
            idx = random.randint(0, array_size - 1)
            addr = base_addr + idx * element_size
            trace.append(addr)
        
        return trace
    
    def strided_access_trace(self, array_size: int, 
                            stride: int,
                            num_passes: int = 1,
                            base_addr: int = 0x50000) -> List[int]:
        """
        Generate trace for strided array access.
        
        Common in scientific computing:
        - Accessing matrix columns
        - Multi-dimensional array slicing
        - Poor cache performance for large strides
        
        Parameters:
        -----------
        array_size : int
            Total array size
        stride : int
            Distance between consecutive accesses
        num_passes : int
            Number of complete passes through the array
        base_addr : int
            Base memory address
            
        Returns:
        --------
        trace : List[int]
            Sequence of memory addresses accessed
        """
        element_size = 4
        trace = []
        
        for _ in range(num_passes):
            for i in range(0, array_size, stride):
                addr = base_addr + i * element_size
                trace.append(addr)
        
        return trace
    
    def mixed_workload_trace(self, size: int = 1000) -> List[int]:
        """
        Generate a realistic mixed workload combining multiple patterns.
        
        Simulates real applications that exhibit various access patterns:
        - 40% sequential access
        - 30% strided access
        - 20% random access
        - 10% hotspot (temporal locality)
        
        Parameters:
        -----------
        size : int
            Approximate number of memory accesses
            
        Returns:
        --------
        trace : List[int]
            Sequence of memory addresses accessed
        """
        trace = []
        
        # Sequential portion (40%)
        seq_size = int(size * 0.4)
        trace.extend(self.sequential_scan_trace(seq_size, base_addr=0x10000))
        
        # Strided portion (30%)
        stride_size = int(size * 0.3)
        trace.extend(self.strided_access_trace(stride_size, stride=8, base_addr=0x20000))
        
        # Random portion (20%)
        random_size = int(size * 0.2)
        trace.extend(self.random_access_trace(1000, random_size, base_addr=0x30000))
        
        # Hotspot portion (10%) - repeatedly access small region
        hotspot_size = int(size * 0.1)
        hotspot_region = 100
        trace.extend(self.random_access_trace(hotspot_region, hotspot_size, base_addr=0x40000))
        
        # Shuffle to mix patterns
        random.shuffle(trace)
        
        return trace
    
    def get_workload_suite(self) -> dict:
        """
        Get a comprehensive suite of workloads for evaluation.
        
        Returns:
        --------
        workloads : dict
            Dictionary mapping workload names to traces
        """
        return {
            'matmul_small': self.matrix_multiplication_trace(32),
            'matmul_medium': self.matrix_multiplication_trace(64),
            'matmul_large': self.matrix_multiplication_trace(128),
            'sort_small': self.quicksort_trace(1000),
            'sort_large': self.quicksort_trace(10000),
            'sequential': self.sequential_scan_trace(10000),
            'random': self.random_access_trace(10000, 10000),
            'stride_8': self.strided_access_trace(10000, stride=8),
            'stride_16': self.strided_access_trace(10000, stride=16),
            'mixed': self.mixed_workload_trace(10000)
        }


if __name__ == "__main__":
    # Demonstrate trace generation
    print("="*60)
    print("Memory Trace Generator - Demonstration")
    print("="*60)
    
    generator = TraceGenerator(seed=42)
    
    # Generate various traces
    workloads = {
        'Matrix Mult (32x32)': generator.matrix_multiplication_trace(32),
        'QuickSort (1000)': generator.quicksort_trace(1000),
        'Sequential (1000)': generator.sequential_scan_trace(1000),
        'Random (1000)': generator.random_access_trace(1000, 1000),
        'Strided (s=8, 1000)': generator.strided_access_trace(1000, stride=8),
    }
    
    print("\nGenerated Workload Statistics:")
    print("-" * 60)
    for name, trace in workloads.items():
        unique_addrs = len(set(trace))
        print(f"{name:25s} | Accesses: {len(trace):6d} | Unique: {unique_addrs:6d}")
    
    print("\n" + "="*60)
    print("Sample addresses from Matrix Multiplication trace:")
    matmul_trace = workloads['Matrix Mult (32x32)']
    print(f"First 20 addresses: {[hex(addr) for addr in matmul_trace[:20]]}")
    print("="*60)
