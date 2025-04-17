#!/usr/bin/env python
"""
Demonstration of text classification using BridgeNLP.

This example shows how to use the HuggingFaceClassificationBridge adapter
to perform text classification and integrate it with spaCy.
"""

import spacy
from spacy import displacy

try:
    from bridgenlp.adapters.hf_classification import HuggingFaceClassificationBridge
except ImportError:
    raise ImportError(
        "Hugging Face dependencies not found. Install with: "
        "pip install bridgenlp[huggingface]"
    )


def main():
    """Run the text classification demo."""
    # Create a spaCy pipeline
    nlp = spacy.load("en_core_web_sm")
    
    # Create a text classification bridge with custom labels
    classification_bridge = HuggingFaceClassificationBridge(
        labels=["politics", "sports", "technology", "entertainment", "business"]
    )
    
    # Process multiple texts with text classification
    texts = [
        "The new iPhone was announced yesterday with revolutionary features.",
        "The team scored three goals in the final minutes of the match.",
        "The stock market reached a new high after the interest rate announcement."
    ]
    
    for text in texts:
        doc = nlp(text)
        
        # Apply text classification
        doc = classification_bridge.from_spacy(doc)
        
        # Print the classification results
        print(f"\nText: {text}")
        print("Classification results:")
        for role in doc._.nlp_bridge_roles:
            print(f"  - {role['label']} (confidence: {role['score']:.2f})")
        
        # Visualize the document
        print("\nDocument visualization:")
        displacy.render(doc, style="dep")


if __name__ == "__main__":
    main()
