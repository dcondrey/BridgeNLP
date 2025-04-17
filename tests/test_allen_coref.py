"""
Tests for the AllenNLPCorefBridge class.
"""

import pytest
import spacy
from unittest.mock import MagicMock, patch

from bridgenlp.result import BridgeResult


# Skip tests if AllenNLP is not installed
allennlp_installed = False
try:
    import allennlp
    import allennlp_models
    allennlp_installed = True
except ImportError:
    pass


@pytest.mark.skipif(not allennlp_installed, reason="AllenNLP not installed")
class TestAllenNLPCorefBridge:
    """Test suite for AllenNLPCorefBridge class."""
    
    @pytest.fixture
    def nlp(self):
        """Create a spaCy pipeline for testing."""
        return spacy.blank("en")
    
    @pytest.fixture
    def mock_predictor(self):
        """Create a mock AllenNLP predictor."""
        mock = MagicMock()
        mock.predict.return_value = {
            "document": ["Julie", "hugged", "David", "because", "she", "missed", "him", "."],
            "clusters": [[[0, 1], [4, 5]], [[2, 3], [6, 7]]]
        }
        return mock
    
    def test_init(self):
        """Test initialization with import error handling."""
        with patch("bridgenlp.adapters.allen_coref.allennlp", None):
            with pytest.raises(ImportError):
                from bridgenlp.adapters.allen_coref import AllenNLPCorefBridge
                AllenNLPCorefBridge()
    
    @patch("allennlp_models.coref.CorefPredictor")
    def test_from_text(self, mock_coref_predictor, mock_predictor, nlp):
        """Test processing text."""
        mock_coref_predictor.from_path.return_value = mock_predictor
        
        from bridgenlp.adapters.allen_coref import AllenNLPCorefBridge
        bridge = AllenNLPCorefBridge()
        
        # Override the predictor with our mock
        bridge._predictor = mock_predictor
        
        result = bridge.from_text("Julie hugged David because she missed him.")
        
        # Check that the predictor was called
        mock_predictor.predict.assert_called_once_with(
            document="Julie hugged David because she missed him."
        )
        
        # Check the result
        assert result.tokens == ["Julie", "hugged", "David", "because", "she", "missed", "him", "."]
        assert result.clusters == [[[0, 1], [4, 5]], [[2, 3], [6, 7]]]
    
    @patch("allennlp_models.coref.CorefPredictor")
    def test_from_tokens(self, mock_coref_predictor, mock_predictor, nlp):
        """Test processing tokens."""
        mock_coref_predictor.from_path.return_value = mock_predictor
        
        from bridgenlp.adapters.allen_coref import AllenNLPCorefBridge
        bridge = AllenNLPCorefBridge()
        
        # Override the predictor with our mock
        bridge._predictor = mock_predictor
        
        tokens = ["Julie", "hugged", "David", "because", "she", "missed", "him", "."]
        result = bridge.from_tokens(tokens)
        
        # Check that the predictor was called with the joined tokens
        mock_predictor.predict.assert_called_once_with(
            document="Julie hugged David because she missed him ."
        )
        
        # Check the result
        assert result.tokens == ["Julie", "hugged", "David", "because", "she", "missed", "him", "."]
        assert result.clusters == [[[0, 1], [4, 5]], [[2, 3], [6, 7]]]
    
    @patch("allennlp_models.coref.CorefPredictor")
    def test_from_spacy(self, mock_coref_predictor, mock_predictor, nlp):
        """Test processing a spaCy Doc."""
        mock_coref_predictor.from_path.return_value = mock_predictor
        
        from bridgenlp.adapters.allen_coref import AllenNLPCorefBridge
        bridge = AllenNLPCorefBridge()
        
        # Override the predictor with our mock
        bridge._predictor = mock_predictor
        
        doc = nlp("Julie hugged David because she missed him.")
        processed_doc = bridge.from_spacy(doc)
        
        # Check that the predictor was called
        mock_predictor.predict.assert_called_once_with(
            document="Julie hugged David because she missed him."
        )
        
        # Check that extensions are registered and values are set
        assert processed_doc._.nlp_bridge_clusters is not None
