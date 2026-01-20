"""Fibonacci Implementation in Python

This module provides multiple implementations of the Fibonacci sequence:
- Recursive: Simple but exponential time complexity O(2^n)
- Iterative: Linear time complexity O(n)
- Memoization: Linear time complexity O(n) with manual caching
- LRU Cache: Linear time complexity O(n) with decorator-based caching
"""

from functools import lru_cache
import time


def fibonacci_recursive(n):
    """Calculate Fibonacci number using simple recursion.
    
    Time Complexity: O(2^n) - exponential
    Space Complexity: O(n) - call stack depth
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n):
    """Calculate Fibonacci number using iteration.
    
    Time Complexity: O(n) - linear
    Space Complexity: O(1) - constant
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr


def fibonacci_memoization(n, memo=None):
    """Calculate Fibonacci number using memoization.
    
    Time Complexity: O(n) - linear
    Space Complexity: O(n) - memo dictionary
    
    Args:
        n: The position in the Fibonacci sequence
        memo: Dictionary to store computed values
        
    Returns:
        The nth Fibonacci number
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)
    return memo[n]


@lru_cache(maxsize=None)
def fibonacci_lru_cache(n):
    """Calculate Fibonacci number using LRU cache decorator.
    
    Time Complexity: O(n) - linear
    Space Complexity: O(n) - cache storage
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    return fibonacci_lru_cache(n - 1) + fibonacci_lru_cache(n - 2)


if __name__ == "__main__":
    print("="*60)
    print("FIBONACCI IMPLEMENTATIONS COMPARISON")
    print("="*60)
    
    test_values = [10, 20, 30]
    
    for n in test_values:
        print(f"\nFibonacci({n}):")
        print(f"  Recursive:        {fibonacci_recursive(n)}")
        print(f"  Iterative:        {fibonacci_iterative(n)}")
        print(f"  Memoization:      {fibonacci_memoization(n)}")
        print(f"  LRU Cache:        {fibonacci_lru_cache(n)}")
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON (n=35)")
    print("="*60)
    
    n = 35
    
    # Recursive
    start = time.time()
    result_recursive = fibonacci_recursive(n)
    time_recursive = time.time() - start
    print(f"Recursive:       {result_recursive} | Time: {time_recursive:.6f}s")
    
    # Iterative
    start = time.time()
    result_iterative = fibonacci_iterative(n)
    time_iterative = time.time() - start
    print(f"Iterative:       {result_iterative} | Time: {time_iterative:.6f}s")
    
    # Memoization
    start = time.time()
    result_memo = fibonacci_memoization(n)
    time_memo = time.time() - start
    print(f"Memoization:     {result_memo} | Time: {time_memo:.6f}s")
    
    # LRU Cache
    fibonacci_lru_cache.cache_clear()
    start = time.time()
    result_lru = fibonacci_lru_cache(n)
    time_lru = time.time() - start
    print(f"LRU Cache:       {result_lru} | Time: {time_lru:.6f}s")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n1. RECURSIVE: Simple but exponential time complexity O(2^n)")
    print("   - Pros: Easy to understand, matches mathematical definition")
    print("   - Cons: Very slow for large n, many redundant calculations")
    print("\n2. ITERATIVE: Linear time complexity O(n)")
    print("   - Pros: Fast, simple, minimal memory usage")
    print("   - Cons: Less intuitive than recursive")
    print("\n3. MEMOIZATION: Linear time complexity O(n)")
    print("   - Pros: Recursive style with caching, avoids redundant calls")
    print("   - Cons: Requires manual memo dictionary management")
    print("\n4. LRU_CACHE: Linear time complexity O(n)")
    print("   - Pros: Elegant decorator-based approach, automatic caching")
    print("   - Cons: Requires Python 3.2+, slight overhead from decorator")
    print("\n" + "="*60)
