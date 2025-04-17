#!/usr/bin/env python
"""
Demonstration of sentiment analysis using BridgeNLP.

This example shows how to use the HuggingFaceSentimentBridge adapter
to perform sentiment analysis and integrate it with spaCy.
"""

import spacy
from spacy import displacy

try:
    from bridgenlp.adapters.hf_sentiment import HuggingFaceSentimentBridge
except ImportError:
    raise ImportError(
        "Hugging Face dependencies not found. Install with: "
        "pip install bridgenlp[huggingface]"
    )


def main():
    """Run the sentiment analysis demo."""
    # Create a spaCy pipeline
    nlp = spacy.load("en_core_web_sm")
    
    # Create a sentiment analysis bridge
    sentiment_bridge = HuggingFaceSentimentBridge()
    
    # Process multiple texts with sentiment analysis
    texts = [
        "I love this product! It's amazing and works perfectly.",
        "This movie was terrible. I hated every minute of it.",
        "The service was okay, but could be better."
    ]
    
    for text in texts:
        doc = nlp(text)
        
        # Apply sentiment analysis
        doc = sentiment_bridge.from_spacy(doc)
        
        # Print the sentiment results
        print(f"\nText: {text}")
        print("Sentiment analysis:")
        for role in doc._.nlp_bridge_roles:
            print(f"  - {role['label']} (confidence: {role['score']:.2f})")
        
        # Visualize the document
        print("\nDocument visualization:")
        displacy.render(doc, style="dep")


if __name__ == "__main__":
    main()
