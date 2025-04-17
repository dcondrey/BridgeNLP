"""
Tests for the SpacyBridgePipe class.
"""

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from typing import List

from bridgenlp.base import BridgeBase
from bridgenlp.pipes.spacy_pipe import SpacyBridgePipe
from bridgenlp.result import BridgeResult


# Mock bridge adapter for testing
class MockBridge(BridgeBase):
    """Mock bridge adapter for testing."""
    
    def from_text(self, text: str) -> BridgeResult:
        """Process text and return a mock result."""
        tokens = text.split()
        return BridgeResult(
            tokens=tokens,
            spans=[(0, 1)],
            clusters=[[(0, 1), (2, 3)]],
            roles=[{"role": "ARG0", "text": tokens[0]}],
            labels=["TEST"] * len(tokens)
        )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """Process tokens and return a mock result."""
        return BridgeResult(
            tokens=tokens,
            spans=[(0, 1)],
            clusters=[[(0, 1), (2, 3)]],
            roles=[{"role": "ARG0", "text": tokens[0]}],
            labels=["TEST"] * len(tokens)
        )
    
    def from_spacy(self, doc: Doc) -> Doc:
        """Process a spaCy Doc and return it with mock results attached."""
        result = BridgeResult(
            tokens=[t.text for t in doc],
            spans=[(0, 1)],
            clusters=[[(0, 1), (2, 3)]],
            roles=[{"role": "ARG0", "text": doc[0].text}],
            labels=["TEST"] * len(doc)
        )
        return result.attach_to_spacy(doc)


class TestSpacyBridgePipe:
    """Test suite for SpacyBridgePipe class."""
    
    @pytest.fixture
    def nlp(self):
        """Create a spaCy pipeline for testing."""
        return spacy.blank("en")
    
    @pytest.fixture
    def mock_bridge(self):
        """Create a mock bridge adapter for testing."""
        return MockBridge()
    
    def test_init(self, mock_bridge):
        """Test initialization."""
        pipe = SpacyBridgePipe(bridge=mock_bridge, name="test_pipe")
        assert pipe.bridge == mock_bridge
        assert pipe.name == "test_pipe"
    
    def test_call(self, nlp, mock_bridge):
        """Test calling the pipe on a Doc."""
        pipe = SpacyBridgePipe(bridge=mock_bridge)
        doc = nlp("This is a test")
        
        # Process the Doc
        processed_doc = pipe(doc)
        
        # Check that extensions are registered and values are set
        assert Doc.has_extension("nlp_bridge_spans")
        assert Doc.has_extension("nlp_bridge_clusters")
        assert Doc.has_extension("nlp_bridge_roles")
        assert Doc.has_extension("nlp_bridge_labels")
        
        assert processed_doc._.nlp_bridge_spans == [(0, 1)]
        assert processed_doc._.nlp_bridge_clusters == [[(0, 1), (2, 3)]]
        assert processed_doc._.nlp_bridge_roles == [{"role": "ARG0", "text": "This"}]
        assert processed_doc._.nlp_bridge_labels == ["TEST", "TEST", "TEST", "TEST"]
    
    def test_factory(self, nlp, mock_bridge):
        """Test the factory function for creating a pipe component."""
        # Register the factory if not already registered
        if "bridgenlp" not in nlp.pipe_factories:
            Language.factory("bridgenlp")(create_bridge_component)
        
        # Add the pipe to the pipeline
        nlp.add_pipe("bridgenlp", config={"bridge": mock_bridge})
        
        # Check that the pipe was added
        assert "bridgenlp" in nlp.pipe_names
        
        # Process a Doc
        doc = nlp("This is a test")
        
        # Check that extensions are registered and values are set
        assert doc._.nlp_bridge_spans == [(0, 1)]
        assert doc._.nlp_bridge_clusters == [[(0, 1), (2, 3)]]
        assert doc._.nlp_bridge_roles == [{"role": "ARG0", "text": "This"}]
        assert doc._.nlp_bridge_labels == ["TEST", "TEST", "TEST", "TEST"]


# Import the factory function to make it available for testing
from bridgenlp.pipes.spacy_pipe import create_bridge_component
