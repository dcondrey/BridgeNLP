"""
Tests for the Pipeline class.
"""

import pytest
import spacy

from bridgenlp.base import BridgeBase
from bridgenlp.config import BridgeConfig
from bridgenlp.pipeline import Pipeline
from bridgenlp.result import BridgeResult


class MockAdapter(BridgeBase):
    """Mock adapter for testing the pipeline."""
    
    def __init__(self, name="mock", config=None):
        super().__init__(config)
        self.name = name
        self.calls = 0
    
    def from_text(self, text):
        self.calls += 1
        return BridgeResult(
            tokens=text.split(),
            spans=[(0, 1)] if self.name == "ner" else [],
            clusters=[[(0, 1), (2, 3)]] if self.name == "coref" else [],
            roles=[{"role": "PRED", "text": "test"}] if self.name == "srl" else [],
            labels=["TEST"] * len(text.split()) if self.name == "classify" else []
        )
    
    def from_tokens(self, tokens):
        self.calls += 1
        return BridgeResult(
            tokens=tokens,
            spans=[(0, 1)] if self.name == "ner" else [],
            clusters=[[(0, 1), (2, 3)]] if self.name == "coref" else [],
            roles=[{"role": "PRED", "text": "test"}] if self.name == "srl" else [],
            labels=["TEST"] * len(tokens) if self.name == "classify" else []
        )
    
    def from_spacy(self, doc):
        self.calls += 1
        result = BridgeResult(
            tokens=[t.text for t in doc],
            spans=[(0, 1)] if self.name == "ner" else [],
            clusters=[[(0, 1), (2, 3)]] if self.name == "coref" else [],
            roles=[{"role": "PRED", "text": "test"}] if self.name == "srl" else [],
            labels=["TEST"] * len(doc) if self.name == "classify" else []
        )
        return result.attach_to_spacy(doc)


class TestPipeline:
    """Test suite for Pipeline class."""
    
    def test_init(self):
        """Test initialization with default values."""
        ner = MockAdapter(name="ner")
        coref = MockAdapter(name="coref")
        
        # Create a pipeline
        pipeline = Pipeline([ner, coref])
        
        # Check that the pipeline has the correct adapters
        assert len(pipeline.adapters) == 2
        assert pipeline.adapters[0] == ner
        assert pipeline.adapters[1] == coref
    
    def test_init_empty(self):
        """Test initialization with empty adapter list."""
        with pytest.raises(ValueError):
            Pipeline([])
    
    def test_from_text(self):
        """Test processing text through the pipeline."""
        # Create mock adapters
        ner = MockAdapter(name="ner")
        coref = MockAdapter(name="coref")
        srl = MockAdapter(name="srl")
        
        # Create a pipeline
        pipeline = Pipeline([ner, coref, srl])
        
        # Process text
        result = pipeline.from_text("This is a test")
        
        # Check the result contains data from all adapters
        assert result.tokens == ["This", "is", "a", "test"]
        assert (0, 1) in result.spans  # From NER
        assert len(result.clusters) == 1  # From coref
        assert len(result.roles) == 1  # From SRL
        
        # Check that all adapters were called
        assert ner.calls == 1
        assert coref.calls == 1
        assert srl.calls == 1
    
    def test_from_tokens(self):
        """Test processing tokens through the pipeline."""
        # Create mock adapters
        ner = MockAdapter(name="ner")
        classify = MockAdapter(name="classify")
        
        # Create a pipeline
        pipeline = Pipeline([ner, classify])
        
        # Process tokens
        tokens = ["This", "is", "a", "test"]
        result = pipeline.from_tokens(tokens)
        
        # Check the result contains data from all adapters
        assert result.tokens == tokens
        assert (0, 1) in result.spans  # From NER
        assert result.labels == ["TEST", "TEST", "TEST", "TEST"]  # From classify
        
        # Check that all adapters were called
        assert ner.calls == 1
        assert classify.calls == 1
    
    def test_from_spacy(self):
        """Test processing a spaCy Doc through the pipeline."""
        # Create mock adapters
        ner = MockAdapter(name="ner")
        coref = MockAdapter(name="coref")
        
        # Create a pipeline
        pipeline = Pipeline([ner, coref])
        
        # Create a spaCy Doc
        nlp = spacy.blank("en")
        doc = nlp("This is a test")
        
        # Process the Doc
        result_doc = pipeline.from_spacy(doc)
        
        # Check that the Doc has attributes from all adapters
        assert result_doc._.nlp_bridge_spans == [(0, 1)]  # From NER
        assert len(result_doc._.nlp_bridge_clusters) == 1  # From coref
        
        # Check that all adapters were called
        assert ner.calls == 1
        assert coref.calls == 1
    
    def test_caching(self):
        """Test result caching."""
        # Create mock adapters
        ner = MockAdapter(name="ner")
        
        # Create a pipeline with caching enabled
        config = BridgeConfig(cache_results=True, cache_size=10)
        pipeline = Pipeline([ner], config)
        
        # Process the same text twice
        result1 = pipeline.from_text("This is a test")
        result2 = pipeline.from_text("This is a test")
        
        # Check that the adapter was only called once
        assert ner.calls == 1
        
        # Check that the results are identical
        assert result1.tokens == result2.tokens
        assert result1.spans == result2.spans
        
        # Check cache metrics
        metrics = pipeline.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 1
        assert metrics["cache_hit_ratio"] == 0.5
    
    def test_metrics(self):
        """Test performance metrics."""
        # Create mock adapters
        ner = MockAdapter(name="ner")
        coref = MockAdapter(name="coref")
        
        # Create a pipeline
        pipeline = Pipeline([ner, coref])
        
        # Process text
        pipeline.from_text("This is a test")
        
        # Get the metrics
        metrics = pipeline.get_metrics()
        
        # Check that the metrics include the basics
        assert "num_calls" in metrics
        assert "total_time" in metrics
        assert "total_tokens" in metrics
        
        # Check that adapter-specific metrics are included
        assert "adapter1_num_calls" in metrics
        assert "adapter2_num_calls" in metrics
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Create mock adapters with cleanup tracking
        class CleanupAdapter(MockAdapter):
            def __init__(self, name="mock", config=None):
                super().__init__(name, config)
                self.cleaned_up = False
            
            def cleanup(self):
                self.cleaned_up = True
        
        ner = CleanupAdapter(name="ner")
        coref = CleanupAdapter(name="coref")
        
        # Create a pipeline
        pipeline = Pipeline([ner, coref])
        
        # Call cleanup
        pipeline.cleanup()
        
        # Check that all adapters were cleaned up
        assert ner.cleaned_up
        assert coref.cleaned_up
        
    def test_conditional_execution(self):
        """Test conditional execution of adapters based on previous results."""
        # Create mock adapters
        ner = MockAdapter(name="ner")
        srl = MockAdapter(name="srl")
        classify = MockAdapter(name="classify")
        
        # Create a pipeline with all three adapters
        pipeline = Pipeline([ner, srl, classify])
        
        # Add a condition: only run classify if NER found an entity
        def condition_fn(result):
            return len(result.spans) > 0
            
        pipeline.add_condition(2, condition_fn)
        
        # Process text - should run all adapters since NER finds an entity
        result = pipeline.from_text("This is a test")
        
        # Check that all adapters were called
        assert ner.calls == 1
        assert srl.calls == 1
        assert classify.calls == 1
        
        # Check that the result contains data from all adapters
        assert (0, 1) in result.spans  # From NER
        assert len(result.roles) == 1  # From SRL
        assert result.labels == ["TEST", "TEST", "TEST", "TEST"]  # From classify
        
        # Create a new pipeline with modified NER that doesn't find entities
        class NoEntityNER(MockAdapter):
            def from_text(self, text):
                self.calls += 1
                return BridgeResult(
                    tokens=text.split(),
                    spans=[],  # No entities found
                    clusters=[],
                    roles=[],
                    labels=[]
                )
                
            def from_tokens(self, tokens):
                self.calls += 1
                return BridgeResult(
                    tokens=tokens,
                    spans=[],  # No entities found
                    clusters=[],
                    roles=[],
                    labels=[]
                )
        
        # Create a new pipeline with the modified NER
        no_entity_ner = NoEntityNER(name="ner")
        srl2 = MockAdapter(name="srl")
        classify2 = MockAdapter(name="classify")
        
        pipeline2 = Pipeline([no_entity_ner, srl2, classify2])
        
        # Add the same condition
        pipeline2.add_condition(2, condition_fn)
        
        # Process text - should skip classify because NER doesn't find an entity
        result = pipeline2.from_text("This is a test")
        
        # Check that only the first two adapters were called
        assert no_entity_ner.calls == 1
        assert srl2.calls == 1
        assert classify2.calls == 0
        
        # Check that the result doesn't contain data from the classify adapter
        assert len(result.spans) == 0  # No entities from NER
        assert len(result.roles) == 1  # From SRL
        assert len(result.labels) == 0  # Classify was skipped
        
    def test_invalid_condition(self):
        """Test error handling for invalid condition registration."""
        # Create a pipeline with two adapters
        ner = MockAdapter(name="ner")
        srl = MockAdapter(name="srl")
        pipeline = Pipeline([ner, srl])
        
        # Define a condition function
        def condition_fn(result):
            return True
        
        # Valid adapter index should work
        pipeline.add_condition(1, condition_fn)
        
        # Invalid adapter indices should raise ValueError
        with pytest.raises(ValueError):
            pipeline.add_condition(0, condition_fn)  # First adapter can't have a condition
            
        with pytest.raises(ValueError):
            pipeline.add_condition(2, condition_fn)  # Out of range
            
        with pytest.raises(ValueError):
            pipeline.add_condition(-1, condition_fn)  # Negative index