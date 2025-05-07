#!/usr/bin/env python
"""
Demonstration of pipeline composition using BridgeNLP.

This example shows how to create a pipeline of multiple adapters
to process text with different NLP tasks in a single pass.
"""

import spacy
from spacy import displacy

try:
    from bridgenlp.pipeline import Pipeline
    from bridgenlp.adapters.spacy_ner import SpacyNERBridge
    from bridgenlp.config import BridgeConfig
except ImportError:
    raise ImportError(
        "BridgeNLP not installed. Install with: pip install -e ."
    )

try:
    from bridgenlp.adapters.hf_sentiment import HuggingFaceSentimentBridge
except ImportError:
    raise ImportError(
        "Hugging Face dependencies not found. Install with: "
        "pip install -e '.[huggingface]'"
    )


def main():
    """Run the pipeline composition demo."""
    print("BridgeNLP Pipeline Composition Demo")
    print("-----------------------------------")
    
    # Create a spaCy pipeline
    print("\nInitializing spaCy...")
    nlp = spacy.load("en_core_web_sm")
    
    # Create individual adapters
    print("Creating NER adapter...")
    ner_bridge = SpacyNERBridge(model_name="en_core_web_sm")
    
    print("Creating sentiment analysis adapter...")
    # Enable caching for better performance
    config = BridgeConfig(cache_results=True, collect_metrics=True)
    sentiment_bridge = HuggingFaceSentimentBridge(
        model_name="distilbert-base-uncased-finetuned-sst-2-english", 
        config=config
    )
    
    # Create a pipeline with both adapters
    print("Creating pipeline with NER and sentiment analysis...")
    pipeline = Pipeline([ner_bridge, sentiment_bridge], config)
    
    # Process a text with the full pipeline
    texts = [
        "Apple is looking at buying U.K. startup for $1 billion. It would be a great acquisition.",
        "Microsoft and Google are also interested, but the deal might not happen. This is disappointing.",
        "The startup has developed innovative AI technology that could revolutionize the market."
    ]
    
    print("\nProcessing texts through the pipeline...\n")
    
    # Process each text
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: {text}")
        
        # Create a spaCy Doc
        doc = nlp(text)
        
        # Process the Doc with our pipeline
        doc = pipeline.from_spacy(doc)
        
        # Extract and print named entities
        print("\nNamed entities:")
        entities = []
        for span_start, span_end in doc._.nlp_bridge_spans:
            label = doc._.nlp_bridge_labels[span_start]
            if label != "O":  # "O" means outside any entity
                entity_text = doc[span_start:span_end].text
                entities.append((label, entity_text))
                print(f"  - {label}: {entity_text}")
        
        # Extract and print sentiment
        print("\nSentiment analysis:")
        sentiment = None
        score = 0.0
        
        # Find the sentiment role in the result
        for role in doc._.nlp_bridge_roles:
            if "label" in role and "score" in role:
                sentiment = role["label"]
                score = role["score"]
                break
        
        if sentiment:
            print(f"  - Sentiment: {sentiment} (confidence: {score:.2f})")
            # Add visual indicator
            if sentiment == "POSITIVE":
                print("  - Mood: 😃")
            else:
                print("  - Mood: 😔")
        
    # Print pipeline performance metrics
    print("\nPipeline performance metrics:")
    metrics = pipeline.get_metrics()
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()