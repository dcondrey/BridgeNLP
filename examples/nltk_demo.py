#!/usr/bin/env python
"""
Demonstration of NLTK integration using BridgeNLP.

This example shows how to use the NLTKBridge adapter
to integrate NLTK's NLP capabilities with spaCy.
"""

import spacy
from spacy import displacy

try:
    from bridgenlp.adapters.nltk_adapter import NLTKBridge
except ImportError:
    raise ImportError(
        "NLTK not found. Install with: pip install nltk"
    )


def main():
    """Run the NLTK integration demo."""
    # Create a spaCy pipeline
    nlp = spacy.blank("en")
    
    # Create an NLTK bridge
    nltk_bridge = NLTKBridge(use_pos=True, use_ner=True)
    
    # Process a text with NLTK
    text = "Apple Inc. is planning to open a new store in New York City next month."
    doc = nlp(text)
    
    # Apply NLTK processing
    doc = nltk_bridge.from_spacy(doc)
    
    # Print the POS tags
    print("Part-of-speech tags (from NLTK):")
    for token, tag in zip(doc, doc._.nlp_bridge_labels):
        print(f"  {token.text}: {tag}")
    
    # Print the named entities
    print("\nNamed entities (from NLTK):")
    for start, end in doc._.nlp_bridge_spans:
        entity_text = doc[start:end].text
        entity_type = doc._.nlp_bridge_labels[start]
        print(f"  {entity_type}: {entity_text}")
    
    # Visualize the document
    print("\nDocument visualization:")
    # Convert our custom entities to spaCy's format for visualization
    ents = []
    for start, end in doc._.nlp_bridge_spans:
        label = doc._.nlp_bridge_labels[start]
        ents.append({"start": start, "end": end, "label": label})
    
    displacy.render({"text": doc.text, "ents": ents}, style="ent", manual=True)


if __name__ == "__main__":
    main()
