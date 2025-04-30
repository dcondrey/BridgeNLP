#!/usr/bin/env python
"""
Demo for using the configuration system with BridgeNLP.

This example shows how to use the configuration system to configure
and initialize bridge adapters.
"""

import json
import os
import sys

# Add the parent directory to the path so we can import bridgenlp
# Need to add the parent of the parent directory to find the bridgenlp package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bridgenlp import BridgeConfig
from bridgenlp.cli import load_bridge


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
    
    # Load a bridge adapter with the configuration
    bridge = load_bridge(loaded_config.model_type, config=loaded_config)
    
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
