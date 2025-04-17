"""
Tests for the HuggingFaceSRLBridge class.
"""

import pytest
import spacy
from unittest.mock import MagicMock, patch

from bridgenlp.result import BridgeResult


# Skip tests if Hugging Face Transformers is not installed
hf_installed = False
try:
    import torch
    import transformers
    hf_installed = True
except ImportError:
    pass


@pytest.mark.skipif(not hf_installed, reason="Hugging Face Transformers not installed")
class TestHuggingFaceSRLBridge:
    """Test suite for HuggingFaceSRLBridge class."""
    
    @pytest.fixture
    def nlp(self):
        """Create a spaCy pipeline for testing."""
        return spacy.blank("en")
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock Hugging Face pipeline."""
        mock = MagicMock()
        mock.return_value = [
            {
                "entity_group": "ARG0",
                "word": "Julie",
                "score": 0.99,
                "start": 0,
                "end": 5
            },
            {
                "entity_group": "V",
                "word": "hugged",
                "score": 0.98,
                "start": 6,
                "end": 12
            },
            {
                "entity_group": "ARG1",
                "word": "David",
                "score": 0.97,
                "start": 13,
                "end": 18
            }
        ]
        return mock
    
    def test_init(self):
        """Test initialization with import error handling."""
        with patch("bridgenlp.adapters.hf_srl.transformers", None):
            with pytest.raises(ImportError):
                from bridgenlp.adapters.hf_srl import HuggingFaceSRLBridge
                HuggingFaceSRLBridge()
    
    @patch("transformers.pipeline")
    def test_from_text(self, mock_hf_pipeline, mock_pipeline, nlp):
        """Test processing text."""
        mock_hf_pipeline.return_value = mock_pipeline
        
        from bridgenlp.adapters.hf_srl import HuggingFaceSRLBridge
        bridge = HuggingFaceSRLBridge()
        
        # Override the pipeline with our mock
        bridge._pipeline = mock_pipeline
        
        result = bridge.from_text("Julie hugged David")
        
        # Check that the pipeline was called
        mock_pipeline.assert_called_once_with("Julie hugged David")
        
        # Check the result
        assert result.tokens == ["Julie", "hugged", "David"]
        assert len(result.roles) == 3
        assert result.roles[0]["role"] == "ARG0"
        assert result.roles[0]["text"] == "Julie"
        assert result.roles[1]["role"] == "V"
        assert result.roles[1]["text"] == "hugged"
        assert result.roles[2]["role"] == "ARG1"
        assert result.roles[2]["text"] == "David"
    
    @patch("transformers.pipeline")
    def test_from_tokens(self, mock_hf_pipeline, mock_pipeline, nlp):
        """Test processing tokens."""
        mock_hf_pipeline.return_value = mock_pipeline
        
        from bridgenlp.adapters.hf_srl import HuggingFaceSRLBridge
        bridge = HuggingFaceSRLBridge()
        
        # Override the pipeline with our mock
        bridge._pipeline = mock_pipeline
        
        tokens = ["Julie", "hugged", "David"]
        result = bridge.from_tokens(tokens)
        
        # Check that the pipeline was called with the joined tokens
        mock_pipeline.assert_called_once_with("Julie hugged David")
        
        # Check the result
        assert result.tokens == ["Julie", "hugged", "David"]
        assert len(result.roles) == 3
    
    @patch("transformers.pipeline")
    def test_from_spacy(self, mock_hf_pipeline, mock_pipeline, nlp):
        """Test processing a spaCy Doc."""
        mock_hf_pipeline.return_value = mock_pipeline
        
        from bridgenlp.adapters.hf_srl import HuggingFaceSRLBridge
        bridge = HuggingFaceSRLBridge()
        
        # Override the pipeline with our mock
        bridge._pipeline = mock_pipeline
        
        doc = nlp("Julie hugged David")
        processed_doc = bridge.from_spacy(doc)
        
        # Check that the pipeline was called
        mock_pipeline.assert_called_once_with("Julie hugged David")
        
        # Check that extensions are registered and values are set
        assert processed_doc._.nlp_bridge_roles is not None
