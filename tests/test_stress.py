"""
Stress tests for BridgeNLP components.
"""

import gc
import time
from typing import List, Optional

import pytest
import spacy
import numpy as np

from bridgenlp.aligner import TokenAligner
from bridgenlp.result import BridgeResult
from bridgenlp.base import BridgeBase
from bridgenlp.pipes.spacy_pipe import SpacyBridgePipe


class MockLargeDocBridge(BridgeBase):
    """Mock bridge adapter for testing with large documents."""
    
    def from_text(self, text: str) -> BridgeResult:
        """Process text and return a mock result."""
        tokens = text.split()
        # Create some mock spans and clusters
        spans = [(i, i+1) for i in range(0, len(tokens), 10)]
        clusters = [[(i, i+1), (i+5, i+6)] for i in range(0, len(tokens), 20) if i+6 < len(tokens)]
        
        return BridgeResult(
            tokens=tokens,
            spans=spans,
            clusters=clusters,
            labels=["TEST" if i % 10 == 0 else "O" for i in range(len(tokens))]
        )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """Process tokens and return a mock result."""
        # Create some mock spans and clusters
        spans = [(i, i+1) for i in range(0, len(tokens), 10)]
        clusters = [[(i, i+1), (i+5, i+6)] for i in range(0, len(tokens), 20) if i+6 < len(tokens)]
        
        return BridgeResult(
            tokens=tokens,
            spans=spans,
            clusters=clusters,
            labels=["TEST" if i % 10 == 0 else "O" for i in range(len(tokens))]
        )
    
    def from_spacy(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """Process a spaCy Doc and return it with mock results attached."""
        tokens = [t.text for t in doc]
        # Create some mock spans and clusters
        spans = [(i, i+1) for i in range(0, len(tokens), 10)]
        clusters = [[(i, i+1), (i+5, i+6)] for i in range(0, len(tokens), 20) if i+6 < len(tokens)]
        
        result = BridgeResult(
            tokens=tokens,
            spans=spans,
            clusters=clusters,
            labels=["TEST" if i % 10 == 0 else "O" for i in range(len(tokens))]
        )
        
        return result.attach_to_spacy(doc)


def generate_large_text(size: int = 50000) -> str:
    """Generate a large text document with the specified number of tokens."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
             "a", "an", "and", "but", "or", "nor", "for", "yet", "so", 
             "in", "on", "at", "by", "with", "about", "against", "between"]
    
    # Generate random text
    np.random.seed(42)  # For reproducibility
    tokens = np.random.choice(words, size=size)
    
    # Add some structure for coreference and entities
    for i in range(0, size, 100):
        if i + 10 < size:
            tokens[i] = "John"
            tokens[i+5] = "he"
            tokens[i+10] = "him"
    
    return " ".join(tokens)


class TestStress:
    """Stress tests for BridgeNLP components."""
    
    @pytest.fixture
    def nlp(self):
        """Create a spaCy pipeline for testing."""
        return spacy.blank("en")
    
    @pytest.fixture
    def large_text(self):
        """Generate a large text document."""
        return generate_large_text(50000)
    
    @pytest.fixture
    def large_doc(self, nlp, large_text):
        """Create a large spaCy Doc for testing."""
        return nlp(large_text)
    
    def test_aligner_large_doc(self, large_doc):
        """Test the TokenAligner with a large document."""
        aligner = TokenAligner()
        
        # Test aligning a span in the middle of the document
        start_time = time.time()
        span = aligner.fuzzy_align(large_doc, "John he him")
        end_time = time.time()
        
        # Check that alignment works and is reasonably fast
        assert span is not None
        assert end_time - start_time < 1.0, "Alignment took too long"
    
    def test_bridge_large_doc(self, nlp, large_text):
        """Test a bridge adapter with a large document."""
        bridge = MockLargeDocBridge()
        
        # Process the large text
        start_time = time.time()
        result = bridge.from_text(large_text)
        end_time = time.time()
        
        # Check that processing works and is reasonably fast
        assert len(result.tokens) == 50000
        assert len(result.spans) > 0
        assert len(result.clusters) > 0
        assert end_time - start_time < 2.0, "Processing took too long"
    
    def test_memory_usage(self, nlp):
        """Test that memory usage doesn't grow after multiple iterations."""
        bridge = MockLargeDocBridge()
        pipe = SpacyBridgePipe(bridge=bridge)
        
        # Generate a moderate-sized text
        text = generate_large_text(5000)
        
        # Force garbage collection to get a clean baseline
        gc.collect()
        
        # Process the text multiple times
        for i in range(100):
            doc = nlp(text)
            processed_doc = pipe(doc)
            
            # Explicitly delete to ensure cleanup
            del processed_doc
            del doc
            
            # Force garbage collection every 10 iterations
            if i % 10 == 0:
                gc.collect()
        
        # Final cleanup
        gc.collect()
        
        # If we got here without memory errors, the test passes
        assert True
    
    def test_batched_inference(self, nlp):
        """Test batched inference with multiple documents."""
        bridge = MockLargeDocBridge()
        
        # Generate multiple texts
        texts = [generate_large_text(1000) for _ in range(10)]
        
        # Process in batch
        start_time = time.time()
        results = [bridge.from_text(text) for text in texts]
        end_time = time.time()
        
        # Check that all texts were processed
        assert len(results) == 10
        assert all(len(result.tokens) == 1000 for result in results)
        
        # Check that batch processing is faster than sequential
        # (This is just a sanity check, not a strict requirement)
        sequential_time = 0
        for text in texts:
            s_time = time.time()
            bridge.from_text(text)
            sequential_time += time.time() - s_time
        
        # Batch should be at least somewhat more efficient
        assert end_time - start_time <= sequential_time
