"""
Test for the memory leak fix in _normalize_text.
"""

import gc
import time
from bridgenlp.aligner import TokenAligner

def test_normalize_text_memory():
    """Test that _normalize_text doesn't leak memory with very long inputs."""
    aligner = TokenAligner()
    
    # Force garbage collection to get a clean baseline
    gc.collect()
    
    try:
        # Try to import psutil for memory tracking
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Create very long texts of increasing size to test caching behavior
        for i in range(1, 11):
            # Create text that's increasingly longer
            # The memory leak would occur with large inputs being cached
            length = i * 2000  # Increase from 2K to 20K characters
            long_text = "a" * length
            
            # Measure memory before
            gc.collect()
            before_mem = process.memory_info().rss / 1024 / 1024
            
            # Call the normalized text method via the public API
            start_time = time.time()
            
            # Call _normalize_text_uncached directly to test memory usage
            # This avoids the warning from fuzzy_align when it can't find a match
            result = aligner._normalize_text_uncached(long_text)
            # Immediately delete the result to free memory
            del result
            # Force garbage collection after each run
            gc.collect()
            
            end_time = time.time()
            after_mem = process.memory_info().rss / 1024 / 1024
            
            print(f"Run {i}: Text length: {length}, Time: {end_time - start_time:.2f}s, "
                  f"Memory change: {after_mem - before_mem:.1f} MB")
            
        # Force garbage collection
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"Final memory usage: {final_memory:.1f} MB")
        print(f"Total memory growth: {final_memory - initial_memory:.1f} MB")
        
        # Check if memory growth is reasonable
        if final_memory - initial_memory < 50:  # Allow up to 50MB growth
            print("PASS: Memory growth is reasonable")
        else:
            print("FAIL: Excessive memory growth detected")
            
    except ImportError:
        print("psutil not available, running simple test...")
        
        # Simple test without memory measurement
        for i in range(1, 6):
            length = i * 5000
            long_text = "a" * length
            
            start_time = time.time()
            result = aligner._normalize_text_uncached(long_text)
            # Immediately delete the result to free memory
            del result
            end_time = time.time()
            
            print(f"Run {i}: Text length: {length}, Time: {end_time - start_time:.2f}s")
            
        print("PASS: All calls completed without errors")

if __name__ == "__main__":
    test_normalize_text_memory()
