#!/usr/bin/env python
"""
Demonstration of prompt conditioning for image captioning using BridgeNLP.

This example showcases how to use prompt conditioning to guide the generation
of image captions, allowing for more specific and controlled descriptions.
"""

import os
import sys
import argparse

# Add the parent directory to the path to allow running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from bridgenlp.adapters.image_captioning import ImageCaptioningBridge
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


def compare_prompt_strategies(image_path: str, use_gpu: bool = False):
    """
    Compare different prompt conditioning strategies for image captioning.
    
    Args:
        image_path: Path to the image file
        use_gpu: Whether to use GPU for processing
    """
    device = 0 if use_gpu else -1
    
    print(f"\nProcessing image: {image_path}")
    print("-" * 60)
    
    # Define test prompts
    prompts = {
        "descriptive": "Describe this image in detail:",
        "emotional": "What emotions does this image convey?",
        "objects": "List the main objects in this image:",
        "setting": "Describe the setting and atmosphere of this image:",
        "storytelling": "Tell a short story based on this image:"
    }
    
    # Define strategies to test
    strategies = ["prefix", "template", "instruction"]
    
    results = {}
    
    # Process with each strategy and prompt
    for strategy in strategies:
        results[strategy] = {}
        
        print(f"\nTesting strategy: {strategy}")
        print("-" * 40)
        
        # Create captioning adapter with this strategy
        config = BridgeConfig()
        config.device = device
        config.modality = "image"
        config.collect_metrics = True
        
        # Create adapter with prompt conditioning enabled
        captioning = ImageCaptioningBridge(
            device=device,
            config=config,
            enable_prompt_conditioning=True,
            prompt_strategy=strategy,
            prompt_template="{prompt}\nCaption: {caption}"  # Custom template for "template" strategy
        )
        
        # Test each prompt with this strategy
        for prompt_name, prompt_text in prompts.items():
            print(f"\nPrompt: {prompt_name}")
            result = captioning.from_image(image_path, prompt=prompt_text)
            caption = result.captions[0]
            results[strategy][prompt_name] = caption
            print(f"Caption: {caption}")
        
        # Clean up resources
        captioning.cleanup()
    
    # Also run once without conditioning for comparison
    print("\nBaseline (No Prompt Conditioning)")
    print("-" * 40)
    
    baseline_captioning = ImageCaptioningBridge(device=device, config=config)
    baseline_result = baseline_captioning.from_image(image_path)
    baseline_caption = baseline_result.captions[0]
    print(f"Caption: {baseline_caption}")
    baseline_captioning.cleanup()
    
    # Display comparison using rich if available
    if HAS_RICH:
        console = Console()
        
        console.print("\n[bold cyan]Prompt Conditioning Comparison[/bold cyan]")
        
        # Create comparison table
        table = Table(title="Image Captioning with Different Prompt Strategies")
        table.add_column("Prompt", style="cyan")
        table.add_column("Prefix Strategy", style="green")
        table.add_column("Template Strategy", style="yellow")
        table.add_column("Instruction Strategy", style="magenta")
        
        for prompt_name in prompts.keys():
            table.add_row(
                prompt_name,
                results["prefix"][prompt_name],
                results["template"][prompt_name],
                results["instruction"][prompt_name]
            )
        
        console.print(table)
        
        # Show baseline comparison
        console.print(Panel(
            f"Without prompt conditioning: {baseline_caption}",
            title="Baseline (No Prompting)",
            border_style="red"
        ))


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="BridgeNLP Prompt Conditioning Demo")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for processing")
    
    args = parser.parse_args()
    
    # Verify the image file exists
    if not os.path.exists(args.image_path) or not os.path.isfile(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)
    
    # Process the image with different prompts
    try:
        compare_prompt_strategies(args.image_path, args.gpu)
    except ImportError as e:
        print(f"Error: Missing dependencies. {str(e)}")
        print("Install required packages with: pip install 'bridgenlp[multimodal]'")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print("Tip: If this is a model loading error, make sure you have internet access")
        print("     and that the specified model is available on Hugging Face Hub.")
        sys.exit(1)


if __name__ == "__main__":
    main()