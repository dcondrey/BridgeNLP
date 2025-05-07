#!/usr/bin/env python
"""
Demonstration of language detection and translation features in BridgeNLP.

This example showcases:
1. Detecting languages of text with confidence scores
2. Getting supported languages for translation models
3. Translating with auto-detection of source language
4. Batch processing for efficient language identification
"""

import os
import sys
import argparse
from pprint import pprint

# Add the parent directory to the path to allow running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from bridgenlp.adapters.hf_translation import HuggingFaceTranslationBridge
    from bridgenlp.config import BridgeConfig
except ImportError:
    print("Error: BridgeNLP package not found. Make sure it's installed or run from the correct directory.")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Note: Install 'rich' package for better output formatting: pip install rich")


def demonstrate_language_detection(translator):
    """
    Demonstrate language detection capabilities.
    
    Args:
        translator: HuggingFaceTranslationBridge instance
    """
    print("\n=== Language Detection Demonstration ===\n")
    
    # Sample texts in different languages
    texts = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "French": "Le vif renard brun saute par-dessus le chien paresseux.",
        "Spanish": "El rápido zorro marrón salta sobre el perro perezoso.",
        "German": "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Italian": "La veloce volpe marrone salta sopra il cane pigro.",
        "Portuguese": "A rápida raposa marrom pula sobre o cachorro preguiçoso.",
        "Dutch": "De snelle bruine vos springt over de luie hond.",
        "Russian": "Быстрая коричневая лиса перепрыгивает через ленивого пса.",
        "Japanese": "速い茶色のキツネは怠け者の犬を飛び越えます。",
        "Chinese": "快速的棕色狐狸跳过懒惰的狗。",
    }
    
    # Detect language for each text
    print("Detecting languages with confidence scores:\n")
    
    if HAS_RICH:
        console = Console()
        table = Table(title="Language Detection Results")
        table.add_column("Text Sample", style="cyan")
        table.add_column("Detected Language", style="green")
        table.add_column("ISO Code", style="yellow")
        table.add_column("Confidence", style="magenta")
        table.add_column("Supported by Model", style="blue")
        
        for language, text in texts.items():
            detection = translator.detect_language(text)
            table.add_row(
                text[:30] + "..." if len(text) > 30 else text,
                detection.name or "Unknown",
                detection.code,
                f"{detection.confidence:.2f}",
                "✓" if detection.supported else "✗"
            )
        
        console.print(table)
    else:
        for language, text in texts.items():
            detection = translator.detect_language(text)
            print(f"Text: {text[:30]}...")
            print(f"  Expected: {language}")
            print(f"  Detected: {detection.name} ({detection.code})")
            print(f"  Confidence: {detection.confidence:.2f}")
            print(f"  Supported: {'Yes' if detection.supported else 'No'}")
            print("")
    
    # Demonstrate batch detection
    print("\nBatch language detection (more efficient):")
    batch_texts = list(texts.values())
    
    batch_start = __import__('time').time()
    batch_detections = translator.detect_language_batch(batch_texts)
    batch_end = __import__('time').time()
    
    # Compare with sequential detection
    seq_start = __import__('time').time()
    seq_detections = [translator.detect_language(text) for text in batch_texts]
    seq_end = __import__('time').time()
    
    print(f"Batch detection time: {batch_end - batch_start:.4f} seconds")
    print(f"Sequential detection time: {seq_end - seq_start:.4f} seconds")
    
    if batch_end - batch_start < seq_end - seq_start:
        print(f"Batch is {(seq_end - seq_start) / (batch_end - batch_start):.1f}x faster!")
    else:
        print("Sequential was faster in this case (small batch size or first-time initialization)")
    
    if HAS_RICH:
        # Example output from batch detection
        console.print("\n[bold]Batch detection results:[/bold]")
        for i, detection in enumerate(batch_detections[:3]):  # Show first 3 for brevity
            console.print(f"{i+1}. {batch_texts[i][:30]}... → [green]{detection.name}[/green] ({detection.code})")


def demonstrate_translation_with_detection(translator):
    """
    Demonstrate translation with automatic language detection.
    
    Args:
        translator: HuggingFaceTranslationBridge instance
    """
    print("\n=== Translation with Language Detection ===\n")
    
    # Enable auto-detection
    translator.set_auto_detect(True)
    
    # Sample texts in different languages
    sample_texts = {
        "English": "The weather is nice today.",
        "French": "J'adore la cuisine française.",
        "Spanish": "Madrid es la capital de España.",
        "German": "Ich lerne seit drei Jahren Deutsch.",
    }
    
    # Translate each text
    print("Translating texts with automatic language detection:\n")
    
    if HAS_RICH:
        console = Console()
        table = Table(title="Translation with Auto-Detection")
        table.add_column("Original Text", style="cyan")
        table.add_column("Detected Language", style="green")
        table.add_column("Translation", style="yellow")
        table.add_column("Confidence", style="magenta")
        
        for language, text in sample_texts.items():
            result = translator.from_text(text, detect_lang=True)
            translation = result.roles[0]["text"]
            detection = result.roles[0]["detection"]
            
            table.add_row(
                text,
                f"{detection['name']} ({detection['code']})",
                translation,
                f"{detection['confidence']:.2f}"
            )
        
        console.print(table)
    else:
        for language, text in sample_texts.items():
            result = translator.from_text(text, detect_lang=True)
            translation = result.roles[0]["text"]
            detection = result.roles[0]["detection"]
            
            print(f"Original ({language}): {text}")
            print(f"Detected: {detection['name']} ({detection['code']})")
            print(f"Translation: {translation}")
            print(f"Confidence: {detection['confidence']:.2f}")
            print("")
    
    # Batch translation with detection
    print("\nBatch translation with detection:")
    batch_texts = list(sample_texts.values())
    batch_results = translator.from_batch(batch_texts, detect_lang=True)
    
    for i, result in enumerate(batch_results):
        original = batch_texts[i]
        translation = result.roles[0]["text"]
        detection = result.roles[0]["detection"]
        
        print(f"{i+1}. {original} → [{detection['code']}] → {translation}")


def demonstrate_supported_languages(translator):
    """
    Demonstrate language support information.
    
    Args:
        translator: HuggingFaceTranslationBridge instance
    """
    print("\n=== Supported Languages ===\n")
    
    # Get supported languages
    languages = translator.get_supported_languages()
    
    # Display information
    print(f"Translation Model: {translator.model_name}")
    print(f"Source Languages: {', '.join(languages['source_languages'])}")
    print(f"Target Languages: {', '.join(languages['target_languages'])}")
    
    # Get detection capabilities
    metrics = translator.get_metrics()
    detection_libraries = metrics.get("language_detection_libraries", [])
    
    print(f"\nLanguage Detection Libraries: {', '.join(detection_libraries)}")
    
    # Show how to get language names
    if HAS_RICH:
        console = Console()
        table = Table(title="Language Code to Name Mapping (Sample)")
        table.add_column("ISO Code", style="cyan")
        table.add_column("Language Name", style="green")
        
        # Show a sample of language codes and names
        for code in ["en", "fr", "de", "es", "it", "ja", "zh", "ru"]:
            name = translator._get_language_name(code) or "Unknown"
            table.add_row(code, name)
        
        console.print(table)


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="BridgeNLP Language Detection Demo")
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-en-fr",
                        help="Translation model to use (default: Helsinki-NLP/opus-mt-en-fr)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for processing")
    parser.add_argument("--demo", choices=["detection", "translation", "supported", "all"],
                        default="all", help="Demo to run (default: all)")
    
    args = parser.parse_args()
    
    print("BridgeNLP Language Detection & Translation Demo")
    print("=" * 50)
    
    try:
        # Create configuration
        config = BridgeConfig()
        config.device = 0 if args.gpu else -1
        config.collect_metrics = True
        
        # Initialize the translation bridge
        translator = HuggingFaceTranslationBridge(
            model_name=args.model,
            config=config
        )
        
        # Print basic information
        print(f"Model: {translator.model_name}")
        print(f"Device: {'GPU' if args.gpu else 'CPU'}")
        
        # Run selected demos
        if args.demo in ["detection", "all"]:
            demonstrate_language_detection(translator)
            
        if args.demo in ["translation", "all"]:
            demonstrate_translation_with_detection(translator)
            
        if args.demo in ["supported", "all"]:
            demonstrate_supported_languages(translator)
        
        # Clean up
        translator.cleanup()
        
    except ImportError as e:
        print(f"Error: Missing dependencies. {str(e)}")
        print("Install required packages with: pip install transformers torch langdetect")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()