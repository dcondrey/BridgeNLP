"""
Test script to debug spaCy integration in Pipeline.
"""

import spacy
from spacy.tokens import Doc
from bridgenlp.base import BridgeBase
from bridgenlp.pipeline import Pipeline
from bridgenlp.result import BridgeResult
from bridgenlp.config import BridgeConfig

class MockAdapter(BridgeBase):
    """Mock adapter for testing the pipeline."""
    
    def __init__(self, name="mock", config=None):
        super().__init__(config)
        self.name = name
        self.calls = 0
    
    def from_text(self, text):
        """Implementation for abstract method."""
        self.calls += 1
        return BridgeResult(
            tokens=text.split(),
            spans=[(0, 1)] if self.name == "ner" else [],
            clusters=[[(0, 1), (2, 3)]] if self.name == "coref" else [],
            roles=[{"role": "PRED", "text": "test"}] if self.name == "srl" else [],
            labels=["TEST"] * len(text.split()) if self.name == "classify" else []
        )
        
    def from_tokens(self, tokens):
        """Implementation for abstract method."""
        self.calls += 1
        return BridgeResult(
            tokens=tokens,
            spans=[(0, 1)] if self.name == "ner" else [],
            clusters=[[(0, 1), (2, 3)]] if self.name == "coref" else [],
            roles=[{"role": "PRED", "text": "test"}] if self.name == "srl" else [],
            labels=["TEST"] * len(tokens) if self.name == "classify" else []
        )
    
    def from_spacy(self, doc):
        """Process a spaCy doc."""
        self.calls += 1
        print(f"Adapter {self.name}: Processing doc")
        
        # Create the result with data based on the adapter type
        result = BridgeResult(
            tokens=[t.text for t in doc],
            spans=[(0, 1)] if self.name == "ner" else [],
            clusters=[[(0, 1), (2, 3)]] if self.name == "coref" else [],
            roles=[{"role": "PRED", "text": "test"}] if self.name == "srl" else [],
            labels=["TEST"] * len(doc) if self.name == "classify" else []
        )
        
        # Attach the result to the doc
        doc = result.attach_to_spacy(doc)
        
        # Print the current state of the doc
        print(f"Adapter {self.name}: Doc state after processing:")
        print(f"  - spans: {doc._.nlp_bridge_spans}")
        print(f"  - clusters: {doc._.nlp_bridge_clusters}")
        print(f"  - roles: {doc._.nlp_bridge_roles}")
        print(f"  - labels: {doc._.nlp_bridge_labels}")
        
        return doc

# Create spaCy doc
nlp = spacy.blank("en")
doc = nlp("This is a test")
print(f"Doc created: {doc}")

# Create adapters
ner = MockAdapter(name="ner")
coref = MockAdapter(name="coref")

# Create pipeline
pipeline = Pipeline([ner, coref])

# Process doc
result_doc = pipeline.from_spacy(doc)

# Check final state
print("\nFinal doc state:")
print(f"  - spans: {result_doc._.nlp_bridge_spans}")
print(f"  - clusters: {result_doc._.nlp_bridge_clusters}")
print(f"  - roles: {result_doc._.nlp_bridge_roles}")
print(f"  - labels: {result_doc._.nlp_bridge_labels}")

# Expected: spans should contain [(0, 1)] from the NER adapter
assert result_doc._.nlp_bridge_spans == [(0, 1)], f"Expected spans: [(0, 1)], got: {result_doc._.nlp_bridge_spans}"

# Expected: clusters should contain [[(0, 1), (2, 3)]] from the Coref adapter
assert result_doc._.nlp_bridge_clusters == [[(0, 1), (2, 3)]], f"Expected clusters: [[(0, 1), (2, 3)]], got: {result_doc._.nlp_bridge_clusters}"

print("\nTest passed successfully!")
