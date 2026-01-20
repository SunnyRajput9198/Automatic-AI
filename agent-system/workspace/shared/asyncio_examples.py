#!/usr/bin/env python3
"""
Asyncio Examples: Comprehensive Working Examples
Demonstrates key asyncio concepts with executable code
"""

import asyncio
import time
from typing import List


# ============================================================================
# EXAMPLE 1: Basic Async Function with Await
# ============================================================================

async def example1_basic_async():
    """
    Demonstrates a simple async function with await.
    Shows how to define and run a basic coroutine.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Async Function with Await")
    print("="*70)
    
    async def fetch_user(user_id: int) -> str:
        """Simulate fetching user data from an API"""
        print(f"  [START] Fetching user {user_id}...")
        await asyncio.sleep(1)  # Simulate I/O operation
        print(f"  [DONE] User {user_id} fetched")
        return f"User-{user_id} data"
    
    # Run the coroutine
    result = await fetch_user(101)
    print(f"  Result: {result}")


# ============================================================================
# EXAMPLE 2: Running Multiple Coroutines with asyncio.gather()
# ============================================================================

async def example2_gather():
    """
    Demonstrates running multiple coroutines concurrently using gather().
    Shows how multiple I/O operations can run in parallel.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Coroutines with asyncio.gather()")
    print("="*70)
    
    async def fetch_data(endpoint: str, delay: float) -> str:
        """Simulate HTTP request to an endpoint"""
        print(f"  [START] Fetching {endpoint}...")
        await asyncio.sleep(delay)  # Simulate network delay
        print(f"  [DONE] {endpoint} completed")
        return f"Data from {endpoint}"
    
    # Create multiple coroutines
    start_time = time.time()
    
    results = await asyncio.gather(
        fetch_data("/api/users", 2),
        fetch_data("/api/posts", 1.5),
        fetch_data("/api/comments", 1),
    )
    
    elapsed = time.time() - start_time
    print(f"\n  All tasks completed in {elapsed:.2f}s")
    print(f"  Results: {results}")
    print(f"  Note: Sequential would take 4.5s, concurrent took {elapsed:.2f}s")


# ============================================================================
# EXAMPLE 3: Using asyncio.create_task()
# ============================================================================

async def example3_create_task():
    """
    Demonstrates using create_task() to schedule coroutines.
    Shows explicit task creation and management.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Using asyncio.create_task()")
    print("="*70)
    
    async def worker(worker_id: int, duration: float) -> str:
        """Simulate a worker performing a task"""
        print(f"  [START] Worker-{worker_id} started (will take {duration}s)")
        await asyncio.sleep(duration)
        print(f"  [DONE] Worker-{worker_id} finished")
        return f"Worker-{worker_id} result"
    
    # Create tasks explicitly
    task1 = asyncio.create_task(worker(1, 2))
    task2 = asyncio.create_task(worker(2, 1.5))
    task3 = asyncio.create_task(worker(3, 1))
    
    print("\n  Tasks created and scheduled for concurrent execution")
    
    # Wait for all tasks to complete
    start_time = time.time()
    results = await asyncio.gather(task1, task2, task3)
    elapsed = time.time() - start_time
    
    print(f"\n  All tasks completed in {elapsed:.2f}s")
    print(f"  Results: {results}")


# ============================================================================
# EXAMPLE 4: Simple Async HTTP-like Operation Simulation
# ============================================================================

async def example4_http_simulation():
    """
    Simulates realistic HTTP operations with error handling and retries.
    Demonstrates practical async patterns for web operations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Async HTTP-like Operation Simulation")
    print("="*70)
    
    async def http_request(url: str, timeout: float = 3) -> dict:
        """Simulate an HTTP request with timeout"""
        print(f"  [REQUEST] GET {url}")
        
        try:
            # Simulate network delay
            await asyncio.sleep(0.5)
            
            # Simulate occasional failures
            if "fail" in url:
                raise ConnectionError(f"Failed to connect to {url}")
            
            print(f"  [SUCCESS] {url} returned 200 OK")
            return {
                "status": 200,
                "url": url,
                "data": f"Response from {url}"
            }
        except ConnectionError as e:
            print(f"  [ERROR] {url} - {e}")
            return {
                "status": 500,
                "url": url,
                "error": str(e)
            }
    
    async def fetch_multiple_endpoints(urls: List[str]) -> List[dict]:
        """Fetch multiple endpoints concurrently"""
        tasks = [http_request(url) for url in urls]
        return await asyncio.gather(*tasks)
    
    # Simulate fetching multiple endpoints
    urls = [
        "https://api.example.com/users",
        "https://api.example.com/posts",
        "https://api.example.com/comments",
    ]
    
    start_time = time.time()
    responses = await fetch_multiple_endpoints(urls)
    elapsed = time.time() - start_time
    
    print(f"\n  Fetched {len(responses)} endpoints in {elapsed:.2f}s")
    print(f"  Successful responses: {sum(1 for r in responses if r['status'] == 200)}")


# ============================================================================
# BONUS EXAMPLE 5: Producer-Consumer Pattern
# ============================================================================

async def example5_producer_consumer():
    """
    Demonstrates the producer-consumer pattern with asyncio.Queue.
    Shows how to coordinate multiple async tasks.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Producer-Consumer Pattern (Bonus)")
    print("="*70)
    
    async def producer(queue: asyncio.Queue, num_items: int):
        """Produce items and put them in the queue"""
        for i in range(num_items):
            item = f"Item-{i}"
            await asyncio.sleep(0.3)  # Simulate production time
            await queue.put(item)
            print(f"  [PRODUCED] {item}")
        
        # Signal completion
        await queue.put(None)
    
    async def consumer(queue: asyncio.Queue, consumer_id: int):
        """Consume items from the queue"""
        while True:
            item = await queue.get()
            if item is None:
                print(f"  [CONSUMER-{consumer_id}] Finished")
                break
            
            await asyncio.sleep(0.2)  # Simulate processing time
            print(f"  [CONSUMER-{consumer_id}] Consumed {item}")
    
    # Create queue and run producer and consumers
    queue = asyncio.Queue()
    
    start_time = time.time()
    await asyncio.gather(
        producer(queue, 5),
        consumer(queue, 1),
        consumer(queue, 2),
    )
    elapsed = time.time() - start_time
    
    print(f"\n  Producer-Consumer completed in {elapsed:.2f}s")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """
    Run all examples sequentially.
    """
    print("\n" + "#"*70)
    print("# ASYNCIO COMPREHENSIVE EXAMPLES")
    print("#"*70)
    
    # Run all examples
    await example1_basic_async()
    await example2_gather()
    await example3_create_task()
    await example4_http_simulation()
    await example5_producer_consumer()
    
    print("\n" + "#"*70)
    print("# ALL EXAMPLES COMPLETED")
    print("#"*70 + "\n")


if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
