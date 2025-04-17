"""
Tests for the TokenAligner class.
"""

import pytest
import spacy

from bridgenlp.aligner import TokenAligner


class TestTokenAligner:
    """Test suite for TokenAligner class."""
    
    @pytest.fixture
    def nlp(self):
        """Create a spaCy pipeline for testing."""
        return spacy.blank("en")
    
    @pytest.fixture
    def aligner(self):
        """Create a TokenAligner instance for testing."""
        return TokenAligner()
    
    def test_align_char_span_exact(self, nlp, aligner):
        """Test aligning character spans that match token boundaries exactly."""
        doc = nlp("This is a test document.")
        
        # Align "This" (0-4)
        span = aligner.align_char_span(doc, 0, 4)
        assert span.text == "This"
        assert span.start == 0
        assert span.end == 1
        
        # Align "test" (10-14)
        span = aligner.align_char_span(doc, 10, 14)
        assert span.text == "test"
        assert span.start == 3
        assert span.end == 4
    
    def test_align_char_span_partial(self, nlp, aligner):
        """Test aligning character spans that don't match token boundaries."""
        doc = nlp("This is a test document.")
        
        # Align "is a" (5-9)
        span = aligner.align_char_span(doc, 5, 9)
        assert span.text == "is a"
        assert span.start == 1
        assert span.end == 3
        
        # Align partial word "te" in "test" (10-12)
        span = aligner.align_char_span(doc, 10, 12)
        assert span.text == "test"  # Should expand to full token
        assert span.start == 3
        assert span.end == 4
    
    def test_align_char_span_invalid(self, nlp, aligner):
        """Test aligning invalid character spans."""
        doc = nlp("This is a test document.")
        
        # Invalid span (end before start)
        span = aligner.align_char_span(doc, 10, 5)
        assert span is None
        
        # Invalid span (out of bounds)
        span = aligner.align_char_span(doc, -5, 5)
        assert span is None
        
        span = aligner.align_char_span(doc, 50, 60)
        assert span is None
    
    def test_align_token_span(self, nlp, aligner):
        """Test aligning token spans from a different tokenization."""
        doc = nlp("This is a test document.")
        
        # Different tokenization: ["This", "is", "a", "test", "document", "."]
        model_tokens = ["This", "is", "a", "test", "document", "."]
        
        # Align "This is" (0-2)
        span = aligner.align_token_span(doc, 0, 2, model_tokens)
        assert span.text == "This is"
        assert span.start == 0
        assert span.end == 2
        
        # Align "test document" (3-5)
        span = aligner.align_token_span(doc, 3, 5, model_tokens)
        assert span.text == "test document"
        assert span.start == 3
        assert span.end == 5
    
    def test_fuzzy_align_exact(self, nlp, aligner):
        """Test fuzzy alignment with exact matches."""
        doc = nlp("This is a test document with some complex structure.")
        
        # Exact match
        span = aligner.fuzzy_align(doc, "test document")
        assert span.text == "test document"
        assert span.start == 3
        assert span.end == 5
    
    def test_fuzzy_align_approximate(self, nlp, aligner):
        """Test fuzzy alignment with approximate matches."""
        doc = nlp("This is a test document with some complex structure.")
        
        # Approximate match (extra spaces, different case)
        span = aligner.fuzzy_align(doc, "  TEST   document  ")
        assert span.text == "test document"
        assert span.start == 3
        assert span.end == 5
    
    def test_fuzzy_align_no_match(self, nlp, aligner):
        """Test fuzzy alignment with no good match."""
        doc = nlp("This is a test document with some complex structure.")
        
        # No good match
        span = aligner.fuzzy_align(doc, "completely different text")
        assert span is None
