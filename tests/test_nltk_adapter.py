"""
Tests for the NLTKBridge class.
"""

import pytest
import spacy
from unittest.mock import MagicMock, patch

from bridgenlp.result import BridgeResult


# Skip tests if NLTK is not installed
nltk_installed = False
try:
    import nltk
    nltk_installed = True
except ImportError:
    pass


@pytest.mark.skipif(not nltk_installed, reason="NLTK not installed")
class TestNLTKBridge:
    """Test suite for NLTKBridge class."""
    
    @pytest.fixture
    def nlp(self):
        """Create a spaCy pipeline for testing."""
        return spacy.blank("en")
    
    def test_init(self):
        """Test initialization with error handling."""
        with patch("bridgenlp.adapters.nltk_adapter.nltk", None):
            with pytest.raises(ImportError):
                from bridgenlp.adapters.nltk_adapter import NLTKBridge
                NLTKBridge()
    
    def test_from_text(self):
        """Test processing text."""
        from bridgenlp.adapters.nltk_adapter import NLTKBridge
        
        bridge = NLTKBridge(use_pos=True, use_ner=True)
        result = bridge.from_text("John Smith works at Apple Inc.")
        
        # Check the result
        assert "John" in result.tokens
        assert "Smith" in result.tokens
        assert "Apple" in result.tokens
        assert len(result.labels) == len(result.tokens)
        
        # Check that we have some POS tags
        assert any(tag != "O" for tag in result.labels)
        
        # Check that we have some named entities
        assert len(result.spans) > 0
    
    def test_from_tokens(self):
        """Test processing tokens."""
        from bridgenlp.adapters.nltk_adapter import NLTKBridge
        
        bridge = NLTKBridge(use_pos=True, use_ner=True)
        tokens = ["John", "Smith", "works", "at", "Apple", "Inc", "."]
        result = bridge.from_tokens(tokens)
        
        # Check the result
        assert result.tokens == tokens
        assert len(result.labels) == len(tokens)
        
        # Check that we have some POS tags
        assert any(tag != "O" for tag in result.labels)
        
        # Check that we have some named entities
        assert len(result.spans) > 0
    
    def test_from_spacy(self, nlp):
        """Test processing a spaCy Doc."""
        from bridgenlp.adapters.nltk_adapter import NLTKBridge
        
        bridge = NLTKBridge(use_pos=True, use_ner=True)
        doc = nlp("John Smith works at Apple Inc.")
        processed_doc = bridge.from_spacy(doc)
        
        # Check that extensions are registered and values are set
        assert processed_doc._.nlp_bridge_labels is not None
        assert processed_doc._.nlp_bridge_spans is not None
        
        # Check that we have some POS tags
        assert any(tag != "O" for tag in processed_doc._.nlp_bridge_labels)
        
        # Check that we have some named entities
        assert len(processed_doc._.nlp_bridge_spans) > 0
