#!/usr/bin/env python
"""
Demo for using the Hugging Face embeddings adapter.

This example shows how to use the HuggingFaceEmbeddingsBridge to generate
embeddings for text and visualize them.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any

# Add the parent directory to the path so we can import bridgenlp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bridgenlp import BridgeConfig
from bridgenlp.adapters.hf_embeddings import HuggingFaceEmbeddingsBridge


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return np.dot(vec1, vec2) / (norm1 * norm2)


def main():
    """Run the embeddings demo."""
    # Create a configuration
    config = BridgeConfig(
        model_type="embeddings",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        collect_metrics=True,
        params={
            "pooling": "mean",
            "normalize": True
        }
    )
    
    # Initialize the bridge
    bridge = HuggingFaceEmbeddingsBridge(config=config)
    
    # Example texts
    texts = [
        "I love machine learning and natural language processing.",
        "Deep learning models have revolutionized NLP.",
        "The weather is beautiful today.",
        "It's a sunny day with clear skies.",
        "Python is my favorite programming language."
    ]
    
    print("Generating embeddings for example texts...")
    
    # Process texts and store embeddings
    embeddings = []
    with bridge:
        for text in texts:
            result = bridge.from_text(text)
            embedding = result.roles[0]["embedding"]
            embeddings.append(embedding)
            print(f"Text: {text}")
            print(f"Embedding shape: {len(embedding)} dimensions")
            print()
    
    # Calculate similarity matrix
    print("Similarity matrix:")
    for i, emb1 in enumerate(embeddings):
        for j, emb2 in enumerate(embeddings):
            sim = cosine_similarity(emb1, emb2)
            print(f"{i+1} vs {j+1}: {sim:.4f}", end="\t")
        print()
    
    # Print performance metrics
    print("\nPerformance metrics:")
    metrics = bridge.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
