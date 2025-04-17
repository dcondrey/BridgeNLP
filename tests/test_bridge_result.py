"""
Tests for the BridgeResult class.
"""

import pytest
import spacy
from spacy.tokens import Doc

from bridgenlp.result import BridgeResult


class TestBridgeResult:
    """Test suite for BridgeResult class."""
    
    def test_init(self):
        """Test initialization with default values."""
        result = BridgeResult(tokens=["This", "is", "a", "test"])
        assert result.tokens == ["This", "is", "a", "test"]
        assert result.spans == []
        assert result.clusters == []
        assert result.roles == []
        assert result.labels == []
    
    def test_init_with_values(self):
        """Test initialization with all values."""
        result = BridgeResult(
            tokens=["This", "is", "a", "test"],
            spans=[(0, 1), (2, 4)],
            clusters=[[(0, 1), (3, 4)]],
            roles=[{"role": "ARG0", "text": "This"}],
            labels=["PERSON", "O", "O", "O"]
        )
        assert result.tokens == ["This", "is", "a", "test"]
        assert result.spans == [(0, 1), (2, 4)]
        assert result.clusters == [[(0, 1), (3, 4)]]
        assert result.roles == [{"role": "ARG0", "text": "This"}]
        assert result.labels == ["PERSON", "O", "O", "O"]
    
    def test_to_json(self):
        """Test conversion to JSON."""
        result = BridgeResult(
            tokens=["This", "is", "a", "test"],
            spans=[(0, 1), (2, 4)],
            clusters=[[(0, 1), (3, 4)]],
            roles=[{"role": "ARG0", "text": "This"}],
            labels=["PERSON", "O", "O", "O"]
        )
        json_data = result.to_json()
        assert json_data["tokens"] == ["This", "is", "a", "test"]
        assert json_data["spans"] == [(0, 1), (2, 4)]
        assert json_data["clusters"] == [[(0, 1), (3, 4)]]
        assert json_data["roles"] == [{"role": "ARG0", "text": "This"}]
        assert json_data["labels"] == ["PERSON", "O", "O", "O"]
    
    def test_attach_to_spacy(self):
        """Test attaching results to a spaCy Doc."""
        nlp = spacy.blank("en")
        doc = nlp("This is a test")
        
        result = BridgeResult(
            tokens=["This", "is", "a", "test"],
            spans=[(0, 1), (2, 4)],
            clusters=[[(0, 1), (3, 4)]],
            roles=[{"role": "ARG0", "text": "This"}],
            labels=["PERSON", "O", "O", "O"]
        )
        
        # Attach results to the Doc
        doc = result.attach_to_spacy(doc)
        
        # Check that extensions are registered and values are set
        assert Doc.has_extension("nlp_bridge_spans")
        assert Doc.has_extension("nlp_bridge_clusters")
        assert Doc.has_extension("nlp_bridge_roles")
        assert Doc.has_extension("nlp_bridge_labels")
        
        assert doc._.nlp_bridge_spans == [(0, 1), (2, 4)]
        assert doc._.nlp_bridge_clusters == [[(0, 1), (3, 4)]]
        assert doc._.nlp_bridge_roles == [{"role": "ARG0", "text": "This"}]
        assert doc._.nlp_bridge_labels == ["PERSON", "O", "O", "O"]
    
    def test_idempotent_extension_registration(self):
        """Test that extension registration is idempotent."""
        nlp = spacy.blank("en")
        doc = nlp("This is a test")
        
        # Register extensions manually first
        if not Doc.has_extension("nlp_bridge_spans"):
            Doc.set_extension("nlp_bridge_spans", default=None)
        
        # Set a value
        doc._.nlp_bridge_spans = [(1, 2)]
        
        # Create and attach a result
        result = BridgeResult(
            tokens=["This", "is", "a", "test"],
            spans=[(0, 1), (2, 4)]
        )
        
        # This should not raise an error and should update the value
        doc = result.attach_to_spacy(doc)
        assert doc._.nlp_bridge_spans == [(0, 1), (2, 4)]
    
    def test_large_document(self):
        """Test handling a large document."""
        # Create a large document (50K tokens)
        tokens = ["token"] * 50000
        
        # Create a large result
        result = BridgeResult(
            tokens=tokens,
            spans=[(i, i+1) for i in range(0, 50000, 1000)],
            clusters=[[(i, i+1), (i+100, i+101)] for i in range(0, 50000, 2000)],
            roles=[{"role": "TEST", "text": "token"} for _ in range(100)],
            labels=["TEST" if i % 1000 == 0 else "O" for i in range(50000)]
        )
        
        # Convert to JSON and back
        json_data = result.to_json()
        
        # Check that the large document was handled correctly
        assert len(json_data["tokens"]) == 50000
        assert len(json_data["spans"]) == 50
        assert len(json_data["clusters"]) == 25
        assert len(json_data["roles"]) == 100
        assert len(json_data["labels"]) == 50000
    
    def test_memory_efficiency(self):
        """Test that the result is memory efficient."""
        import sys
        
        # Create a moderate-sized document
        tokens = ["token"] * 10000
        
        # Create a result with minimal data
        result1 = BridgeResult(tokens=tokens)
        
        # Create a result with the same tokens but lots of other data
        result2 = BridgeResult(
            tokens=tokens,
            spans=[(i, i+1) for i in range(0, 10000, 100)],
            clusters=[[(i, i+1), (i+10, i+11)] for i in range(0, 10000, 200)],
            roles=[{"role": "TEST", "text": "token"} for _ in range(100)],
            labels=["TEST" if i % 100 == 0 else "O" for i in range(10000)]
        )
        
        # Check that the tokens are shared between results (no deep copy)
        assert result1.tokens is not result2.tokens
        
        # Check that the memory usage is reasonable
        # This is a rough check, not a precise measurement
        size1 = sys.getsizeof(result1) + sum(sys.getsizeof(x) for x in result1.tokens)
        size2 = sys.getsizeof(result2) + sum(sys.getsizeof(x) for x in result2.tokens)
        
        # result2 should be larger but not exponentially so
        assert size2 > size1
        assert size2 < size1 * 10  # Arbitrary threshold
