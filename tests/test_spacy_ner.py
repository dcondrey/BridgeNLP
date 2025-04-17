"""
Tests for the SpacyNERBridge class.
"""

import pytest
import spacy
from unittest.mock import MagicMock, patch

from bridgenlp.adapters.spacy_ner import SpacyNERBridge
from bridgenlp.result import BridgeResult


class TestSpacyNERBridge:
    """Test suite for SpacyNERBridge class."""
    
    @pytest.fixture
    def nlp(self):
        """Create a spaCy pipeline for testing."""
        return spacy.blank("en")
    
    @pytest.fixture
    def mock_nlp(self):
        """Create a mock spaCy pipeline with NER."""
        mock = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = [
            MagicMock(start=0, end=1, label_="PERSON", text="Julie"),
            MagicMock(start=2, end=3, label_="PERSON", text="David")
        ]
        mock_doc.__len__ = lambda self: 7
        mock_doc.text = "Julie hugged David because she missed him."
        
        # Set up tokens
        tokens = []
        for i, text in enumerate(["Julie", "hugged", "David", "because", "she", "missed", "him", "."]):
            token = MagicMock()
            token.text = text
            token.idx = sum(len(t) + 1 for t in ["Julie", "hugged", "David", "because", "she", "missed", "him", "."][:i])
            tokens.append(token)
        
        mock_doc.__iter__ = lambda self: iter(tokens)
        mock_doc.__getitem__ = lambda self, idx: tokens[idx] if isinstance(idx, int) else tokens[idx.start:idx.stop]
        
        mock.return_value = mock_doc
        mock.pipe_names = ["ner"]
        
        return mock
    
    def test_init(self):
        """Test initialization with error handling."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = OSError("Model not found")
            with pytest.raises(ImportError):
                SpacyNERBridge(model_name="nonexistent_model")
    
    def test_from_text(self, mock_nlp):
        """Test processing text."""
        with patch("spacy.load", return_value=mock_nlp):
            bridge = SpacyNERBridge()
            
            # Override the nlp with our mock
            bridge.nlp = mock_nlp
            
            result = bridge.from_text("Julie hugged David because she missed him.")
            
            # Check that the nlp was called
            mock_nlp.assert_called_once_with("Julie hugged David because she missed him.")
            
            # Check the result
            assert len(result.spans) == 2
            assert "PERSON" in result.labels
    
    def test_from_tokens(self, mock_nlp):
        """Test processing tokens."""
        with patch("spacy.load", return_value=mock_nlp):
            bridge = SpacyNERBridge()
            
            # Override the nlp with our mock
            bridge.nlp = mock_nlp
            
            tokens = ["Julie", "hugged", "David", "because", "she", "missed", "him", "."]
            result = bridge.from_tokens(tokens)
            
            # Check that the nlp was called with the joined tokens
            mock_nlp.assert_called_once_with("Julie hugged David because she missed him .")
            
            # Check the result
            assert len(result.spans) == 2
            assert "PERSON" in result.labels
    
    def test_from_spacy(self, nlp, mock_nlp):
        """Test processing a spaCy Doc."""
        with patch("spacy.load", return_value=mock_nlp):
            bridge = SpacyNERBridge()
            
            # Override the nlp with our mock
            bridge.nlp = mock_nlp
            
            doc = nlp("Julie hugged David because she missed him.")
            processed_doc = bridge.from_spacy(doc)
            
            # Check that the nlp was called
            mock_nlp.assert_called_once_with("Julie hugged David because she missed him.")
            
            # Check that extensions are registered and values are set
            assert processed_doc._.nlp_bridge_spans is not None
            assert processed_doc._.nlp_bridge_labels is not None
            assert "PERSON" in processed_doc._.nlp_bridge_labels
