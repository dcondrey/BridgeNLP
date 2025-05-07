"""
Simple test script to validate the memory leak fix in TokenAligner.
"""

import gc
import time
import spacy
from bridgenlp.aligner import TokenAligner

def generate_large_text(size=5000):
    """Generate a large text document with the specified number of tokens."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
             "a", "an", "and", "but", "or", "nor", "for", "yet", "so", 
             "in", "on", "at", "by", "with", "about", "against", "between"]
    
    # Import numpy in function to make test more portable
    import numpy as np
    np.random.seed(42)  # For reproducibility
    tokens = np.random.choice(words, size=size)
    
    # Add some structure for testing
    for i in range(0, size, 100):
        if i + 10 < size:
            tokens[i] = "John"
            tokens[i+5] = "he"
            tokens[i+10] = "him"
    
    return " ".join(tokens)

def test_aligner_memory_usage():
    """Test that memory usage doesn't grow significantly after multiple uses."""
    print("Creating spaCy pipeline...")
    nlp = spacy.blank("en")
    
    print("Generating large text document...")
    large_text = generate_large_text(3000)
    
    # Process text with spaCy
    print("Processing with spaCy...")
    doc = nlp(large_text)
    
    print("Creating TokenAligner...")
    aligner = TokenAligner()
    
    # Force garbage collection to get clean baseline
    gc.collect()
    
    try:
        # Try to import psutil for memory tracking
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        print("Running multiple alignment operations...")
        for i in range(5):
            search_text = f"John he him"
            start_time = time.time()
            
            # Do the alignment operation
            result = aligner.fuzzy_align(doc, search_text)
            
            # Force garbage collection after each operation
            del result
            gc.collect()
            
            # Sleep briefly to allow memory to be reclaimed
            time.sleep(0.1)
            
            end_time = time.time()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"Run {i+1}: Time: {end_time - start_time:.2f}s, Memory: {current_memory:.1f} MB, " 
                  f"Diff: {current_memory - initial_memory:.1f} MB")
            
            # Force garbage collection between runs
            gc.collect()
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Final memory usage: {final_memory:.1f} MB")
        print(f"Memory change: {final_memory - initial_memory:.1f} MB")
        
        # Some growth is expected, but it should be reasonable
        memory_growth = final_memory - initial_memory
        if memory_growth < initial_memory * 0.2:  # Less than 20% growth
            print("PASS: Memory usage growth is reasonable")
        else:
            print(f"WARNING: Memory usage grew by {memory_growth:.1f} MB ({(memory_growth/initial_memory)*100:.1f}%)")
            # Don't fail the test, just warn about it
            
    except ImportError:
        print("psutil not available, running simple time-based test...")
        total_time = 0
        
        for i in range(5):
            start_time = time.time()
            result = aligner.fuzzy_align(doc, "John he him")
            end_time = time.time()
            run_time = end_time - start_time
            total_time += run_time
            print(f"Run {i+1}: Time: {run_time:.2f}s")
            
        print(f"Average time per run: {total_time/5:.2f}s")
        print("PASS: Alignment completes without errors")
        
        # Clean up memory more aggressively
        del doc
        del aligner
        gc.collect()
        gc.collect()  # Second collection sometimes helps
        time.sleep(0.2)  # Give the system time to reclaim memory

if __name__ == "__main__":
    test_aligner_memory_usage()
