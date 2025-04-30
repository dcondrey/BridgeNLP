#!/usr/bin/env python
"""
Demo for using the configuration system with BridgeNLP.

This example shows how to use the configuration system to configure
and initialize bridge adapters.
"""

import json
import os
import sys
from typing import List, Optional

# Add the parent directory to the path so we can import bridgenlp
# Need to add the parent of the parent directory to find the bridgenlp package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bridgenlp import BridgeConfig
from bridgenlp.cli import load_bridge
import spacy


def main():
    """Run the configuration demo."""
    # Create a configuration programmatically
    config = BridgeConfig(
        model_type="sentiment",
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        device="cpu",
        batch_size=4,
        collect_metrics=True
    )
    
    # Save the configuration to a file
    config_path = os.path.join(os.path.dirname(__file__), "generated_config.json")
    config.to_json(config_path)
    print(f"Saved configuration to {config_path}")
    
    # Load the configuration from the file
    loaded_config = BridgeConfig.from_json(config_path)
    print("Loaded configuration:")
    print(json.dumps(loaded_config.to_dict(), indent=2))
    
    # Create a mock bridge adapter for the demo
    # This avoids dependency on actual Hugging Face models
    from bridgenlp.base import BridgeBase
    from bridgenlp.result import BridgeResult
    
    class MockSentimentBridge(BridgeBase):
        """Mock sentiment bridge for demo purposes."""
        
        def __init__(self, model_name=None, config=None):
            super().__init__(config)
            self.model_name = model_name or "mock-sentiment"
            print(f"Initialized mock bridge with model: {self.model_name}")
            
        def from_text(self, text: str) -> BridgeResult:
            # Use the context manager properly
            with self._measure_performance():
                # Simple sentiment logic - just for demo
                positive_words = ["love", "amazing", "good", "great", "excellent"]
                negative_words = ["worst", "bad", "terrible", "awful", "hate"]
                
                text_lower = text.lower()
                tokens = text.split()
                
                # Count positive and negative words
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                # Determine sentiment
                if pos_count > neg_count:
                    label = "POSITIVE"
                elif neg_count > pos_count:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                
                # Update metrics
                self._metrics["total_tokens"] += len(tokens)
                
                return BridgeResult(
                    tokens=tokens,
                    labels=[label]
                )
        
        def from_tokens(self, tokens: List[str]) -> BridgeResult:
            with self._measure_performance():
                text = " ".join(tokens)
                # Don't call from_text which would double-count metrics
                # Simple sentiment logic - just for demo
                positive_words = ["love", "amazing", "good", "great", "excellent"]
                negative_words = ["worst", "bad", "terrible", "awful", "hate"]
                
                text_lower = text.lower()
                
                # Count positive and negative words
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                # Determine sentiment
                if pos_count > neg_count:
                    label = "POSITIVE"
                elif neg_count > pos_count:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                
                # Update metrics
                self._metrics["total_tokens"] += len(tokens)
                
                return BridgeResult(
                    tokens=tokens,
                    labels=[label]
                )
        
        def from_spacy(self, doc) -> spacy.tokens.Doc:
            with self._measure_performance():
                result = self.from_text(doc.text)
                return result.attach_to_spacy(doc)
    
    # Use our mock bridge instead of trying to load a real model
    bridge = MockSentimentBridge(model_name=loaded_config.model_name, config=loaded_config)
    
    # Process some text
    texts = [
        "I love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The service was okay, nothing special."
    ]
    
    print("\nProcessing texts:")
    for text in texts:
        result = bridge.from_text(text)
        print(f"Text: {text}")
        print(f"Result: {json.dumps(result.to_json(), indent=2)}")
        print()
    
    # Print performance metrics
    print("Performance metrics:")
    metrics = bridge.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
