#!/usr/bin/env python
"""
Demonstration of conditional pipeline execution using BridgeNLP.

This example shows how to create a pipeline with conditional execution
based on the results of previous adapters.
"""

import spacy

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
    from bridgenlp.adapters.hf_srl import HuggingFaceSRLBridge
except ImportError:
    raise ImportError(
        "Hugging Face dependencies not found. Install with: "
        "pip install -e '.[huggingface]'"
    )


def main():
    """Run the conditional pipeline execution demo."""
    print("BridgeNLP Conditional Pipeline Demo")
    print("-----------------------------------")
    
    # Create a spaCy pipeline
    print("\nInitializing spaCy...")
    nlp = spacy.load("en_core_web_sm")
    
    # Create individual adapters
    print("Creating NER adapter...")
    ner_bridge = SpacyNERBridge(model_name="en_core_web_sm")
    
    print("Creating SRL adapter...")
    srl_bridge = HuggingFaceSRLBridge()
    
    print("Creating sentiment analysis adapter...")
    # Enable caching for better performance
    config = BridgeConfig(cache_results=True, collect_metrics=True)
    sentiment_bridge = HuggingFaceSentimentBridge(
        model_name="distilbert-base-uncased-finetuned-sst-2-english", 
        config=config
    )
    
    # Create a pipeline with all adapters
    print("Creating pipeline with NER, SRL, and sentiment analysis...")
    pipeline = Pipeline([ner_bridge, srl_bridge, sentiment_bridge], config)
    
    # Add a condition: only run sentiment analysis if NER found an organization
    # This allows focusing sentiment analysis only on texts mentioning organizations
    print("Adding condition: Only run sentiment analysis if text mentions an organization...")
    
    def has_organization(result):
        for start, end in result.spans:
            if start < len(result.labels) and result.labels[start] == "ORG":
                return True
        return False
    
    # The sentiment adapter is at index 2 (third adapter)
    pipeline.add_condition(2, has_organization)
    
    # Example texts
    texts = [
        "Apple is looking at buying U.K. startup for $1 billion. It would be a great acquisition.",  # Has ORG
        "The weather in New York is beautiful today. I'm having a wonderful time.",  # No ORG
        "Microsoft announced record profits last quarter, exceeding analysts' expectations.",  # Has ORG
        "She walked her dog in the park yesterday afternoon and enjoyed the sunshine."  # No ORG
    ]
    
    print("\nProcessing texts through the conditional pipeline...\n")
    
    # Process each text
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: {text}")
        
        # Create a spaCy Doc
        doc = nlp(text)
        
        # Process the Doc with our pipeline
        doc = pipeline.from_spacy(doc)
        
        # Extract and print named entities
        print("\nNamed entities:")
        has_org = False
        entities = []
        for span_start, span_end in doc._.nlp_bridge_spans:
            label = doc._.nlp_bridge_labels[span_start]
            if label != "O":  # "O" means outside any entity
                entity_text = doc[span_start:span_end].text
                entities.append((label, entity_text))
                print(f"  - {label}: {entity_text}")
                if label == "ORG":
                    has_org = True
        
        if not entities:
            print("  - No entities found")
        
        # Extract and print semantic roles
        print("\nSemantic roles:")
        if hasattr(doc._, "nlp_bridge_roles") and doc._.nlp_bridge_roles:
            for role in doc._.nlp_bridge_roles:
                if "role" in role and "text" in role:
                    print(f"  - {role['role']}: {role['text']}")
        else:
            print("  - No semantic roles found")
        
        # Extract and print sentiment (if run)
        print("\nSentiment analysis:")
        if has_org:
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
            else:
                print("  - Sentiment analysis was run but no result found")
        else:
            print("  - Skipped (no organization mentioned)")
    
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