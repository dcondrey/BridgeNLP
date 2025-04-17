#!/usr/bin/env python
"""
Demonstration of semantic role labeling using BridgeNLP.

This example shows how to use the HuggingFaceSRLBridge adapter
to perform semantic role labeling and integrate it with spaCy.
"""

import spacy
from spacy import displacy

try:
    from bridgenlp.adapters.hf_srl import HuggingFaceSRLBridge
except ImportError:
    raise ImportError(
        "Hugging Face dependencies not found. Install with: "
        "pip install bridgenlp[huggingface]"
    )


def main():
    """Run the semantic role labeling demo."""
    # Create a spaCy pipeline
    nlp = spacy.load("en_core_web_sm")
    
    # Create a semantic role labeling bridge
    srl_bridge = HuggingFaceSRLBridge()
    
    # Process a text with semantic role labeling
    text = "Julie hugged David because she missed him."
    doc = nlp(text)
    
    # Apply semantic role labeling
    doc = srl_bridge.from_spacy(doc)
    
    # Print the semantic roles
    print("Semantic roles:")
    for role in doc._.nlp_bridge_roles:
        print(f"  - {role['role']}: {role['text']} (score: {role['score']:.2f})")
    
    # Visualize the document
    print("\nDocument visualization:")
    displacy.render(doc, style="dep")


if __name__ == "__main__":
    main()
