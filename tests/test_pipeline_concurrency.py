"""
Tests to verify thread safety in the Pipeline component.
"""

import threading
import time
import random
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

from bridgenlp.base import BridgeBase
from bridgenlp.config import BridgeConfig
from bridgenlp.pipeline import Pipeline
from bridgenlp.result import BridgeResult


class DelayedAdapter(BridgeBase):
    """Mock adapter that introduces random delays to simulate concurrency issues."""
    
    def __init__(self, name="mock", max_delay=0.05, config=None):
        super().__init__(config)
        self.name = name
        self.max_delay = max_delay
        self.calls = 0
        self.call_lock = threading.RLock()
    
    def from_text(self, text):
        with self.call_lock:
            self.calls += 1
            call_number = self.calls
            
        # Simulate variable processing time
        delay = random.uniform(0, self.max_delay)
        time.sleep(delay)
        
        # Simulate processing
        result = BridgeResult(
            tokens=text.split(),
            spans=[(0, 1)] if self.name == "ner" else [],
            clusters=[[(0, 1), (2, 3)]] if self.name == "coref" else [],
            roles=[{"role": "PRED", "text": f"test_{call_number}_{self.name}"}] if self.name == "srl" else [],
            labels=["TEST"] * len(text.split()) if self.name == "classify" else []
        )
        
        return result
    
    def from_tokens(self, tokens):
        with self.call_lock:
            self.calls += 1
            call_number = self.calls
            
        # Simulate variable processing time
        delay = random.uniform(0, self.max_delay)
        time.sleep(delay)
        
        # Simulate processing
        result = BridgeResult(
            tokens=tokens,
            spans=[(0, 1)] if self.name == "ner" else [],
            clusters=[[(0, 1), (2, 3)]] if self.name == "coref" else [],
            roles=[{"role": "PRED", "text": f"test_{call_number}_{self.name}"}] if self.name == "srl" else [],
            labels=["TEST"] * len(tokens) if self.name == "classify" else []
        )
        
        return result
    
    def from_spacy(self, doc):
        with self.call_lock:
            self.calls += 1
            call_number = self.calls
            
        # Simulate variable processing time
        delay = random.uniform(0, self.max_delay)
        time.sleep(delay)
        
        # Simulate processing
        tokens = [t.text for t in doc]
        result = BridgeResult(
            tokens=tokens,
            spans=[(0, 1)] if self.name == "ner" else [],
            clusters=[[(0, 1), (2, 3)]] if self.name == "coref" else [],
            roles=[{"role": "PRED", "text": f"test_{call_number}_{self.name}"}] if self.name == "srl" else [],
            labels=["TEST"] * len(tokens) if self.name == "classify" else []
        )
        
        return result.attach_to_spacy(doc)


def test_concurrent_pipeline_execution():
    """Test concurrent execution of the Pipeline with multiple threads."""
    # Create mock adapters
    ner = DelayedAdapter(name="ner")
    coref = DelayedAdapter(name="coref")
    srl = DelayedAdapter(name="srl")
    
    # Create a pipeline with caching enabled and metrics enabled
    config = BridgeConfig(cache_results=True, cache_size=50, collect_metrics=True)
    pipeline = Pipeline([ner, coref, srl], config)
    
    # Number of concurrent threads
    num_threads = 10
    
    # Number of iterations per thread
    num_iterations = 20
    
    # Define a worker function that runs a pipeline operation
    def worker(worker_id):
        results = []
        for i in range(num_iterations):
            # Use different inputs to test various cache scenarios
            if i % 3 == 0:
                # Text that should hit the cache after first access
                text = f"This is a test message {i % 5}"
            else:
                # Unique text to miss the cache
                text = f"This is a unique test message {worker_id}_{i}"
                
            # Process the text
            result = pipeline.from_text(text)
            results.append(result)
            
            # Occasionally get metrics to test concurrent metrics access
            if random.random() < 0.2:
                metrics = pipeline.get_metrics()
                assert isinstance(metrics, dict)
                
        return results
    
    # Run the workers concurrently
    results_by_thread = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker, i): i for i in range(num_threads)}
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                results_by_thread[worker_id] = future.result()
            except Exception as e:
                print(f"Worker {worker_id} raised an exception: {e}")
                raise
    
    # Verify that all threads completed successfully
    assert len(results_by_thread) == num_threads
    
    # Verify each thread processed the correct number of inputs
    for worker_id, results in results_by_thread.items():
        assert len(results) == num_iterations
        
    # Check that cache was used (should have fewer calls than iterations)
    total_iterations = num_threads * num_iterations
    total_calls = ner.calls + coref.calls + srl.calls
    expected_calls = total_iterations * 3  # 3 adapters
    
    # If caching worked, we should have fewer calls than expected
    assert total_calls < expected_calls, f"Expected fewer than {expected_calls} calls, got {total_calls}"
    
    # Verify metrics are consistent
    metrics = pipeline.get_metrics()
    assert metrics["num_calls"] > 0
    assert metrics["cache_hits"] + metrics["cache_misses"] == total_iterations
    
    # Cleanup
    pipeline.cleanup()
    
    print(f"Completed test_concurrent_pipeline_execution:")
    print(f"- Threads: {num_threads}")
    print(f"- Iterations per thread: {num_iterations}")
    print(f"- Total calls: {total_calls} (vs {expected_calls} max)")
    print(f"- Cache hits: {metrics['cache_hits']}")
    print(f"- Cache misses: {metrics['cache_misses']}")
    print(f"- Cache hit ratio: {metrics['cache_hit_ratio']:.2f}")


def test_concurrent_condition_updates():
    """Test concurrent updates to pipeline conditions."""
    # Create mock adapters
    ner = DelayedAdapter(name="ner")
    coref = DelayedAdapter(name="coref")
    srl = DelayedAdapter(name="srl")
    
    # Create a pipeline with metrics enabled
    config = BridgeConfig(collect_metrics=True)
    pipeline = Pipeline([ner, coref, srl], config)
    
    # Create condition functions
    def condition1(result):
        return len(result.spans) > 0
        
    def condition2(result):
        return len(result.clusters) > 0
    
    # Define a worker function that adds and uses conditions
    def worker(worker_id):
        # Add conditions
        if worker_id % 2 == 0:
            # Even workers set conditions
            pipeline.add_condition(1, condition1)
            time.sleep(0.01)  # Introduce some delay
            pipeline.add_condition(2, condition2)
        else:
            # Odd workers set conditions in reverse order
            pipeline.add_condition(2, condition2)
            time.sleep(0.01)  # Introduce some delay
            pipeline.add_condition(1, condition1)
            
        # Process text through the pipeline
        text = f"This is a test message from worker {worker_id}"
        result = pipeline.from_text(text)
        
        return result
    
    # Run concurrent workers
    num_workers = 5
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker, i): i for i in range(num_workers)}
        results = {}
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                results[worker_id] = future.result()
            except Exception as e:
                print(f"Worker {worker_id} raised an exception: {e}")
                raise
    
    # Verify that all workers completed successfully
    assert len(results) == num_workers
    
    # Verify the conditions were applied
    assert 1 in pipeline._conditions
    assert 2 in pipeline._conditions
    
    # Test the final state of the pipeline
    result = pipeline.from_text("This is a final test")
    
    # Cleanup
    pipeline.cleanup()
    
    print("Completed test_concurrent_condition_updates without thread safety issues")


def test_concurrent_cache_stress():
    """Test concurrent cache operations with high contention."""
    # Create mock adapters
    ner = DelayedAdapter(name="ner")
    coref = DelayedAdapter(name="coref")
    
    # Create a pipeline with a very small cache
    config = BridgeConfig(cache_results=True, cache_size=5, collect_metrics=True)
    pipeline = Pipeline([ner, coref], config)
    
    # Number of concurrent threads
    num_threads = 20
    
    # Number of iterations per thread - using small cache but many iterations
    # will cause high contention on the cache
    num_iterations = 30
    
    # Define a worker function 
    def worker(worker_id):
        for i in range(num_iterations):
            # Create a different input string each time to test cache eviction
            # but reuse some values to test cache hits
            if i % 7 == 0:  # Occasional repeats to test cache hits
                text = f"Common text {i % 5}"
            else:
                text = f"Unique text {worker_id}_{i}"
                
            # Process through pipeline
            result = pipeline.from_text(text)
            
            # Occasionally check metrics or get cache info
            if random.random() < 0.1:
                metrics = pipeline.get_metrics()
                assert isinstance(metrics, dict)
                
        return worker_id
    
    # Run workers concurrently
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker, i): i for i in range(num_threads)}
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Worker {worker_id} raised an exception: {e}")
                raise
    
    # Verify all workers completed
    assert len(results) == num_threads
    
    # Get final metrics
    metrics = pipeline.get_metrics()
    
    # Cleanup
    pipeline.cleanup()
    
    print(f"Completed test_concurrent_cache_stress:")
    print(f"- Threads: {num_threads}")
    print(f"- Iterations per thread: {num_iterations}")
    print(f"- Cache hits: {metrics['cache_hits']}")
    print(f"- Cache misses: {metrics['cache_misses']}")
    print(f"- Cache hit ratio: {metrics['cache_hit_ratio']:.2f}")

if __name__ == "__main__":
    # Run the tests
    test_concurrent_pipeline_execution()
    test_concurrent_condition_updates()
    test_concurrent_cache_stress()
    print("All concurrent tests passed!")
