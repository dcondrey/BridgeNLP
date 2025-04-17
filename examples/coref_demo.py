#!/usr/bin/env python
"""
Demonstration of coreference resolution using BridgeNLP.

This example shows how to use the AllenNLPCorefBridge adapter
to perform coreference resolution and integrate it with spaCy.
"""

import spacy
from spacy import displacy

try:
    from bridgenlp.adapters.allen_coref import AllenNLPCorefBridge
except ImportError:
    raise ImportError(
        "AllenNLP dependencies not found. Install with: "
        "pip install bridgenlp[allennlp]"
    )


def main():
    """Run the coreference resolution demo."""
    # Create a spaCy pipeline
    nlp = spacy.load("en_core_web_sm")
    
    # Create a coreference resolution bridge
    coref_bridge = AllenNLPCorefBridge()
    
    # Process a text with coreference resolution
    text = "Julie hugged David because she missed him. She had not seen him in a long time."
    doc = nlp(text)
    
    # Apply coreference resolution
    doc = coref_bridge.from_spacy(doc)
    
    # Print the coreference clusters
    print("Coreference clusters:")
    for i, cluster in enumerate(doc._.nlp_bridge_clusters):
        print(f"Cluster {i + 1}:")
        for start, end in cluster:
            print(f"  - {doc[start:end].text}")
    
    # Visualize the document
    print("\nDocument visualization:")
    displacy.render(doc, style="dep")


if __name__ == "__main__":
    main()
