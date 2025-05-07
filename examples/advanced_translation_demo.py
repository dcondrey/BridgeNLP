#!/usr/bin/env python
"""
Demonstration of advanced translation capabilities using BridgeNLP.

This example showcases the enhanced translation adapter with:
- Automatic language detection
- Batch processing
- Multiple language support
- Model sharing across adapters
- Performance metrics
"""

import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from bridgenlp.adapters.hf_translation import HuggingFaceTranslationBridge
    from bridgenlp.config import BridgeConfig
except ImportError:
    raise ImportError(
        "BridgeNLP not installed or missing dependencies. "
        "Install with: pip install -e '.[huggingface]'"
    )


def main():
    """Run the advanced translation demo."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]BridgeNLP Advanced Translation Demo[/bold cyan]\n"
        "Showcasing language detection, batch processing, and model sharing",
        border_style="cyan"
    ))
    
    # Sample multilingual texts
    texts = [
        # English
        "Artificial intelligence is transforming the way we live and work.",
        
        # Spanish
        "La inteligencia artificial está transformando nuestra forma de vivir y trabajar.",
        
        # French
        "L'intelligence artificielle transforme notre façon de vivre et de travailler.",
        
        # German
        "Künstliche Intelligenz verändert unsere Art zu leben und zu arbeiten.",
        
        # Italian
        "L'intelligenza artificiale sta trasformando il nostro modo di vivere e lavorare."
    ]
    
    # Step 1: Create translation adapter with auto-detection
    console.print("\n[bold]Step 1: Creating translation adapter with auto-detection[/bold]")
    
    # Create configuration with performance tracking
    config = BridgeConfig(
        collect_metrics=True,
        cache_results=True,
        batch_size=2
    )
    
    try:
        # Create adapter for translating to English
        translator_to_en = HuggingFaceTranslationBridge(
            model_name="Helsinki-NLP/opus-mt-mul-en",  # Multilingual to English
            auto_detect_language=True,  # Auto-detect the source language
            target_lang="en",
            config=config
        )
        console.print("[green]✓[/green] Created multilingual to English translator")
        
        # Create adapter for translating to Spanish
        translator_to_es = HuggingFaceTranslationBridge(
            model_name="Helsinki-NLP/opus-mt-en-es",  # English to Spanish
            source_lang="en",
            target_lang="es",
            config=config
        )
        console.print("[green]✓[/green] Created English to Spanish translator")
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("[yellow]Tip: Install transformers with:[/yellow] pip install transformers torch langdetect")
        return
    
    # Step 2: Automatic language detection and translation to English
    console.print("\n[bold]Step 2: Auto-detecting languages and translating to English[/bold]")
    
    # Create a results table
    table = Table(title="Auto-Detected Translations to English")
    table.add_column("Original", style="yellow")
    table.add_column("Detected Language", style="cyan")
    table.add_column("English Translation", style="green")
    
    try:
        # Process texts as a batch for better performance
        start_time = time.time()
        results = translator_to_en.from_batch(texts)
        batch_time = time.time() - start_time
        
        for original, result in zip(texts, results):
            # Extract information from the result
            detected_lang = result.roles[0]["source_lang"] if result.roles else "unknown"
            translation = result.roles[0]["text"] if result.roles else "Translation failed"
            
            # Add to the table
            table.add_row(original, detected_lang, translation)
        
        console.print(table)
        console.print(f"[dim]Batch translation completed in {batch_time:.2f} seconds[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error during translation:[/bold red] {str(e)}")
    
    # Step 3: English to Spanish translation with all English texts
    console.print("\n[bold]Step 3: Translating detected English texts to Spanish[/bold]")
    
    # Get English texts from previous results
    english_texts = []
    english_originals = []
    
    for i, result in enumerate(results):
        if result.roles and result.roles[0]["source_lang"] == "en":
            english_texts.append(texts[i])
            english_originals.append(texts[i])
        elif result.roles:
            # For non-English originals, use the English translation
            english_texts.append(result.roles[0]["text"])
            english_originals.append(texts[i])
    
    if not english_texts:
        console.print("[yellow]No English texts detected for translation to Spanish[/yellow]")
    else:
        try:
            # Create a results table
            table = Table(title="English to Spanish Translations")
            table.add_column("Original", style="yellow")
            table.add_column("English", style="cyan")
            table.add_column("Spanish Translation", style="green")
            
            # Process texts as a batch
            start_time = time.time()
            results_es = translator_to_es.from_batch(english_texts)
            batch_time = time.time() - start_time
            
            for original, english, result in zip(english_originals, english_texts, results_es):
                # Extract the Spanish translation
                spanish = result.roles[0]["text"] if result.roles else "Translation failed"
                
                # Add to the table
                table.add_row(original, english, spanish)
            
            console.print(table)
            console.print(f"[dim]Batch translation completed in {batch_time:.2f} seconds[/dim]")
            
        except Exception as e:
            console.print(f"[bold red]Error during translation:[/bold red] {str(e)}")
    
    # Step 4: Display performance metrics
    console.print("\n[bold]Step 4: Performance Metrics[/bold]")
    
    # Get metrics from the translators
    metrics_en = translator_to_en.get_metrics()
    metrics_es = translator_to_es.get_metrics()
    
    # Create a metrics table
    table = Table(title="Translation Performance Metrics")
    table.add_column("Metric", style="blue")
    table.add_column("Multi→English", style="cyan")
    table.add_column("English→Spanish", style="green")
    
    # Add core metrics
    metrics_to_show = [
        ("Calls", "num_calls"),
        ("Total tokens", "total_tokens"),
        ("Total characters", "total_characters"),
        ("Batch calls", "batch_calls"),
        ("Memory usage (MB)", "memory_usage_mb"),
        ("Language detections", "language_detections"),
        ("Auto-detect enabled", "auto_detect_enabled"),
    ]
    
    for label, key in metrics_to_show:
        en_value = metrics_en.get(key, "N/A")
        es_value = metrics_es.get(key, "N/A")
        
        # Format values
        if isinstance(en_value, float):
            en_value = f"{en_value:.2f}"
        if isinstance(es_value, float):
            es_value = f"{es_value:.2f}"
            
        table.add_row(label, str(en_value), str(es_value))
    
    console.print(table)
    
    # Step 5: Clean up
    console.print("\n[bold]Step 5: Cleaning up resources[/bold]")
    translator_to_en.cleanup()
    translator_to_es.cleanup()
    console.print("[green]✓[/green] Resources cleaned up")
    

if __name__ == "__main__":
    main()