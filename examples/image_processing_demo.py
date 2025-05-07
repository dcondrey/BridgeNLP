#!/usr/bin/env python
"""
Demonstration of image processing capabilities using BridgeNLP.

This example showcases the multimodal adapters:
- Image captioning
- Object detection
- Combining multiple image analysis adapters
"""

import os
import sys
import argparse
from typing import List, Dict, Any

# Add the parent directory to the path to allow running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from bridgenlp.adapters.image_captioning import ImageCaptioningBridge
    from bridgenlp.adapters.object_detection import ObjectDetectionBridge
    from bridgenlp.pipeline import Pipeline
    from bridgenlp.config import BridgeConfig
except ImportError:
    print("Error: BridgeNLP package not found. Make sure it's installed or run from the correct directory.")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Note: Install 'rich' package for better output formatting: pip install rich")


def process_image(image_path: str, use_gpu: bool = False) -> None:
    """
    Process an image using both captioning and object detection.
    
    Args:
        image_path: Path to the image file
        use_gpu: Whether to use GPU for processing
    """
    device = 0 if use_gpu else -1
    
    # Create a configuration
    config = BridgeConfig()
    config.device = device
    config.modality = "image"
    config.collect_metrics = True
    
    print(f"\nProcessing image: {image_path}")
    print("-" * 50)
    
    # Image captioning
    print("\n1. Image Captioning")
    captioning = ImageCaptioningBridge(device=device, config=config)
    captioning.num_captions = 3  # Generate multiple captions
    
    caption_result = captioning.from_image(image_path)
    
    print("Captions:")
    for i, caption in enumerate(caption_result.captions):
        print(f"  {i+1}. {caption}")
    
    # Object detection
    print("\n2. Object Detection")
    detection = ObjectDetectionBridge(device=device, config=config)
    detection.threshold = 0.7  # Lower threshold to detect more objects
    
    detection_result = detection.from_image(image_path)
    
    print(f"Detected {len(detection_result.detected_objects)} objects:")
    for obj in detection_result.detected_objects:
        print(f"  - {obj['label']} (confidence: {obj['score']:.2f})")
    
    # Display performance metrics
    print("\nPerformance Metrics:")
    print(f"  Captioning: {captioning.get_metrics()['avg_time']:.4f} seconds")
    print(f"  Detection: {detection.get_metrics()['avg_time']:.4f} seconds")
    
    # Clean up resources
    captioning.cleanup()
    detection.cleanup()
    
    # If rich is available, show a more detailed view
    if HAS_RICH:
        console = Console()
        
        console.print("\n[bold cyan]Rich Formatted Output[/bold cyan]")
        
        # Captions panel
        console.print(Panel(
            "\n".join([f"{i+1}. {caption}" for i, caption in enumerate(caption_result.captions)]),
            title="Image Captions",
            border_style="green"
        ))
        
        # Object detection table
        table = Table(title="Detected Objects")
        table.add_column("Label", style="cyan")
        table.add_column("Confidence", style="magenta")
        table.add_column("Bounding Box", style="blue")
        
        for obj in detection_result.detected_objects:
            table.add_row(
                obj['label'],
                f"{obj['score']:.2f}",
                f"{obj['box']}"
            )
        
        console.print(table)


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="BridgeNLP Image Processing Demo")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for processing")
    
    args = parser.parse_args()
    
    # Verify the image file exists
    if not os.path.exists(args.image_path) or not os.path.isfile(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)
    
    # Process the image
    try:
        process_image(args.image_path, args.gpu)
    except ImportError as e:
        print(f"Error: Missing dependencies. {str(e)}")
        print("Install required packages with: pip install 'bridgenlp[multimodal]' or pip install transformers torch pillow")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print("Tip: If this is a model loading error, make sure you have internet access")
        print("     and that the specified model is available on Hugging Face Hub.")
        sys.exit(1)


if __name__ == "__main__":
    main()