#!/usr/bin/env python
"""
Demonstration of multimodal capabilities using BridgeNLP.

This example showcases the integration of text and image processing:
- Text-image embeddings and similarity
- Combining multiple modalities in a pipeline
- Image search by text description
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Tuple

# Add the parent directory to the path to allow running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from bridgenlp.adapters.image_captioning import ImageCaptioningBridge
    from bridgenlp.adapters.object_detection import ObjectDetectionBridge
    from bridgenlp.adapters.multimodal_embeddings import MultimodalEmbeddingsBridge
    from bridgenlp.adapters.hf_sentiment import HuggingFaceSentimentBridge
    from bridgenlp.pipeline import Pipeline
    from bridgenlp.config import BridgeConfig
except ImportError:
    print("Error: BridgeNLP package not found. Make sure it's installed or run from the correct directory.")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Note: Install 'rich' package for better output formatting: pip install rich")


def calculate_text_image_similarity(text: str, image_path: str, use_gpu: bool = False) -> None:
    """
    Calculate and display the similarity between text and image.
    
    Args:
        text: Text to compare
        image_path: Path to the image file
        use_gpu: Whether to use GPU for processing
    """
    device = 0 if use_gpu else -1
    
    # Create a configuration
    config = BridgeConfig()
    config.device = device
    config.modality = "multimodal"
    config.collect_metrics = True
    
    print(f"\nCalculating similarity between text and image")
    print(f"Text: \"{text}\"")
    print(f"Image: {image_path}")
    print("-" * 50)
    
    # Create multimodal embeddings bridge
    embeddings = MultimodalEmbeddingsBridge(device=device, config=config)
    
    # Calculate similarity
    similarity = embeddings.calculate_similarity(text, image_path)
    print(f"\nSimilarity score: {similarity:.4f} (0-1 scale)")
    
    # Get embeddings for visualization
    result = embeddings.from_text_and_image(text, image_path)
    
    # Display some information about the embeddings
    text_embedding = result.roles[0]["text_embedding"]
    image_embedding = result.image_features["embedding"]
    print(f"Text embedding dimension: {len(text_embedding)}")
    print(f"Image embedding dimension: {len(image_embedding)}")
    
    # Clean up resources
    embeddings.cleanup()


def image_search_by_text(text_query: str, image_dir: str, use_gpu: bool = False) -> None:
    """
    Search for images by text description.
    
    Args:
        text_query: Text description to search for
        image_dir: Directory containing images
        use_gpu: Whether to use GPU for processing
    """
    device = 0 if use_gpu else -1
    
    # Create a configuration
    config = BridgeConfig()
    config.device = device
    config.modality = "multimodal"
    
    print(f"\nSearching for images matching: \"{text_query}\"")
    print(f"Image directory: {image_dir}")
    print("-" * 50)
    
    # Find image files in the directory
    image_files = []
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_files.append(os.path.join(image_dir, file))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to search")
    
    # Create multimodal embeddings bridge
    embeddings = MultimodalEmbeddingsBridge(device=device, config=config)
    
    # Calculate text embedding once
    text_result = embeddings.from_text(text_query)
    text_embedding = text_result.roles[0]["embedding"]
    
    # Calculate similarity with each image
    similarities = []
    
    # Use rich progress bar if available
    if HAS_RICH:
        console = Console()
        with Progress() as progress:
            task = progress.add_task("[cyan]Calculating similarities...", total=len(image_files))
            
            for image_path in image_files:
                try:
                    # Get image embedding
                    image_result = embeddings.from_image(image_path)
                    image_embedding = image_result.image_features["embedding"]
                    
                    # Calculate cosine similarity (dot product of normalized vectors)
                    import numpy as np
                    similarity = np.dot(text_embedding, image_embedding)
                    
                    similarities.append((image_path, similarity))
                    progress.update(task, advance=1)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    progress.update(task, advance=1)
    else:
        # Simple progress without rich
        for i, image_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}", end="\r")
            try:
                # Get image embedding
                image_result = embeddings.from_image(image_path)
                image_embedding = image_result.image_features["embedding"]
                
                # Calculate cosine similarity
                import numpy as np
                similarity = np.dot(text_embedding, image_embedding)
                
                similarities.append((image_path, similarity))
            except Exception as e:
                print(f"\nError processing {image_path}: {str(e)}")
        print()  # New line after progress
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Display top results
    print("\nTop matches:")
    for i, (image_path, similarity) in enumerate(similarities[:5]):
        print(f"{i+1}. {os.path.basename(image_path)} - Similarity: {similarity:.4f}")
    
    # Clean up resources
    embeddings.cleanup()


def analyze_image_sentiment(image_path: str, use_gpu: bool = False) -> None:
    """
    Generate a caption for an image and analyze its sentiment.
    
    Args:
        image_path: Path to the image file
        use_gpu: Whether to use GPU for processing
    """
    device = 0 if use_gpu else -1
    
    # Create a configuration
    config = BridgeConfig()
    config.device = device
    config.collect_metrics = True
    
    print(f"\nAnalyzing image sentiment: {image_path}")
    print("-" * 50)
    
    # Create the pipeline with captioning and sentiment analysis
    captioning = ImageCaptioningBridge(device=device)
    sentiment = HuggingFaceSentimentBridge(device=device)
    
    # Create a pipeline that first generates a caption and then analyzes its sentiment
    pipeline = Pipeline([captioning, sentiment], config=config)
    
    # Process the image
    try:
        # Since the pipeline expects text input, we can't use it directly with an image
        # Instead, we'll manually get the caption and then process it
        caption_result = captioning.from_image(image_path)
        caption = caption_result.captions[0]
        
        print(f"Generated caption: \"{caption}\"")
        
        # Analyze sentiment of the caption
        sentiment_result = sentiment.from_text(caption)
        
        # Extract sentiment label and score
        sentiment_label = sentiment_result.labels[0]
        sentiment_score = sentiment_result.roles[0]["score"] if sentiment_result.roles else 0.0
        
        print(f"Sentiment: {sentiment_label} (confidence: {sentiment_score:.4f})")
        
        # Pretty display with rich if available
        if HAS_RICH:
            console = Console()
            
            console.print("\n[bold cyan]Image Sentiment Analysis[/bold cyan]")
            
            console.print(Panel(
                f"Caption: \"{caption}\"\n\nSentiment: [bold]{sentiment_label}[/bold] (confidence: {sentiment_score:.4f})",
                title="Analysis Results",
                border_style="green"
            ))
    
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
    
    # Clean up resources
    captioning.cleanup()
    sentiment.cleanup()


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="BridgeNLP Multimodal Demo")
    parser.add_argument("--image", help="Path to an image file for similarity or sentiment analysis")
    parser.add_argument("--text", help="Text query for similarity calculation or image search")
    parser.add_argument("--dir", help="Directory of images to search through")
    parser.add_argument("--mode", choices=["similarity", "search", "sentiment"], 
                        default="similarity", help="Demo mode to run")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for processing")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "similarity":
            if not args.image or not args.text:
                print("Error: --image and --text are required for similarity mode")
                sys.exit(1)
                
            if not os.path.exists(args.image) or not os.path.isfile(args.image):
                print(f"Error: Image file not found at {args.image}")
                sys.exit(1)
                
            calculate_text_image_similarity(args.text, args.image, args.gpu)
            
        elif args.mode == "search":
            if not args.text or not args.dir:
                print("Error: --text and --dir are required for search mode")
                sys.exit(1)
                
            if not os.path.exists(args.dir) or not os.path.isdir(args.dir):
                print(f"Error: Directory not found at {args.dir}")
                sys.exit(1)
                
            image_search_by_text(args.text, args.dir, args.gpu)
            
        elif args.mode == "sentiment":
            if not args.image:
                print("Error: --image is required for sentiment mode")
                sys.exit(1)
                
            if not os.path.exists(args.image) or not os.path.isfile(args.image):
                print(f"Error: Image file not found at {args.image}")
                sys.exit(1)
                
            analyze_image_sentiment(args.image, args.gpu)
    
    except ImportError as e:
        print(f"Error: Missing dependencies. {str(e)}")
        print("Install required packages with: pip install 'bridgenlp[multimodal]' or pip install transformers torch pillow")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Tip: If this is a model loading error, make sure you have internet access")
        print("     and that the specified model is available on Hugging Face Hub.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, print help
        print("BridgeNLP Multimodal Demo")
        print("\nExample usage:")
        print("  Calculate text-image similarity:")
        print("    python multimodal_demo.py --mode similarity --image path/to/image.jpg --text \"A cat sitting on a couch\"")
        print("\n  Search for images by text description:")
        print("    python multimodal_demo.py --mode search --dir path/to/images/ --text \"beach sunset\"")
        print("\n  Analyze image sentiment:")
        print("    python multimodal_demo.py --mode sentiment --image path/to/image.jpg")
        print("\nUse --gpu flag to enable GPU acceleration.")
        sys.exit(0)
        
    main()