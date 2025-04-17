#!/usr/bin/env python
"""
Demonstration of named entity recognition using BridgeNLP.

This example shows how to use the SpacyNERBridge adapter
to perform named entity recognition and integrate it with spaCy.
"""

import spacy
from spacy import displacy

from bridgenlp.adapters.spacy_ner import SpacyNERBridge


def main():
    """Run the named entity recognition demo."""
    # Create a spaCy pipeline
    nlp = spacy.blank("en")
    
    # Create a named entity recognition bridge
    ner_bridge = SpacyNERBridge(model_name="en_core_web_sm")
    
    # Process a text with named entity recognition
    text = "Apple is looking at buying U.K. startup for $1 billion. Microsoft and Google are also interested."
    doc = nlp(text)
    
    # Apply named entity recognition
    doc = ner_bridge.from_spacy(doc)
    
    # Print the named entities
    print("Named entities:")
    for i, label in enumerate(doc._.nlp_bridge_labels):
        if label != "O":
            span_indices = [(start, end) for start, end in doc._.nlp_bridge_spans 
                           if start <= i < end]
            if span_indices:
                start, end = span_indices[0]
                print(f"  - {label}: {doc[start:end].text}")
    
    # Visualize the document with entities
    print("\nDocument visualization:")
    # Convert our custom entities to spaCy's format for visualization
    ents = []
    for start, end in doc._.nlp_bridge_spans:
        label = doc._.nlp_bridge_labels[start]
        ents.append({"start": start, "end": end, "label": label})
    
    displacy.render({"text": doc.text, "ents": ents}, style="ent", manual=True)


if __name__ == "__main__":
    main()
