#!/usr/bin/env python
"""
Command-line interface for BridgeNLP.

This module provides a command-line tool for using BridgeNLP adapters
without writing Python code.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, TextIO, Union
from io import UnsupportedOperation
from tqdm import tqdm

import spacy

from .base import BridgeBase
from .config import BridgeConfig
from .result import BridgeResult


def load_bridge(model_type: str, model_name: Optional[str] = None, 
                config: Optional[BridgeConfig] = None) -> BridgeBase:
    """
    Load a bridge adapter based on model type and name.
    
    Args:
        model_type: Type of model to load (e.g., 'coref', 'srl', 'ner', 'sentiment', 'classify', 'qa')
        model_name: Optional specific model name
        config: Optional configuration object
        
    Returns:
        Configured bridge adapter
        
    Raises:
        ImportError: If required dependencies are not installed
        ValueError: If model type is not recognized
    """
    # Create default config if none provided
    if config is None:
        config = BridgeConfig(
            model_type=model_type,
            model_name=model_name
        )
    elif model_name is not None and model_name != config.model_name:
        # Override model_name in config if explicitly provided and different
        config.model_name = model_name
    if model_type == "coref" or model_type == "spanbert-coref":
        try:
            from .adapters.allen_coref import AllenNLPCorefBridge
            return AllenNLPCorefBridge(
                model_name=config.model_name or "coref-spanbert",
                config=config
            )
        except ImportError:
            raise ImportError(
                "AllenNLP dependencies not found. Install with: "
                "pip install bridgenlp[allennlp]"
            )
    
    elif model_type == "srl" or model_type == "bert-srl":
        try:
            from .adapters.hf_srl import HuggingFaceSRLBridge
            return HuggingFaceSRLBridge(
                model_name=config.model_name or "Davlan/bert-base-multilingual-cased-srl-nli",
                config=config
            )
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install bridgenlp[huggingface]"
            )
    
    elif model_type == "ner" or model_type == "spacy-ner":
        try:
            from .adapters.spacy_ner import SpacyNERBridge
            
            # First check if the model is available
            model_name = config.model_name or "en_core_web_sm"
            try:
                # Try to load the model to verify it's installed
                import spacy
                spacy.load(model_name)
            except OSError:
                # Model not found, suggest downloading it
                print(f"spaCy model '{model_name}' not found. Attempting to download...", file=sys.stderr)
                try:
                    # Try to download the model automatically
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                    print(f"Successfully downloaded spaCy model: {model_name}", file=sys.stderr)
                except subprocess.CalledProcessError:
                    # If download fails, provide instructions
                    raise ImportError(
                        f"Could not automatically download spaCy model '{model_name}'. "
                        f"Please install it manually with: python -m spacy download {model_name}"
                    )
            
            # Now create the bridge with the verified model
            return SpacyNERBridge(
                model_name=model_name,
                config=config
            )
        except ImportError as e:
            raise ImportError(
                f"Error loading spaCy NER model: {str(e)}. "
                f"Make sure the model is installed with: "
                f"python -m spacy download {config.model_name or 'en_core_web_sm'}"
            )
    
    elif model_type == "sentiment":
        try:
            from .adapters.hf_sentiment import HuggingFaceSentimentBridge
            return HuggingFaceSentimentBridge(
                model_name=config.model_name or "distilbert-base-uncased-finetuned-sst-2-english",
                config=config
            )
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install bridgenlp[huggingface]"
            )
    
    elif model_type == "classify" or model_type == "classification":
        try:
            from .adapters.hf_classification import HuggingFaceClassificationBridge
            return HuggingFaceClassificationBridge(
                model_name=config.model_name or "facebook/bart-large-mnli",
                config=config
            )
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install bridgenlp[huggingface]"
            )
    
    elif model_type == "qa" or model_type == "question-answering":
        try:
            from .adapters.hf_qa import HuggingFaceQABridge
            qa_bridge = HuggingFaceQABridge(
                model_name=config.model_name or "deepset/roberta-base-squad2",
                config=config
            )
            return qa_bridge
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install bridgenlp[huggingface]"
            )
    
    elif model_type == "nltk":
        try:
            from .adapters.nltk_adapter import NLTKBridge
            return NLTKBridge(config=config)
        except ImportError:
            raise ImportError(
                "NLTK not found. Install with: pip install nltk"
            )
    
    elif model_type == "embeddings" or model_type == "embedding":
        try:
            from .adapters.hf_embeddings import HuggingFaceEmbeddingsBridge
            return HuggingFaceEmbeddingsBridge(
                model_name=config.model_name or "sentence-transformers/all-MiniLM-L6-v2",
                config=config
            )
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install transformers torch numpy"
            )
    
    elif model_type == "summarization" or model_type == "summary":
        try:
            from .adapters.hf_summarization import HuggingFaceSummarizationBridge
            return HuggingFaceSummarizationBridge(
                model_name=config.model_name or "facebook/bart-large-cnn",
                config=config
            )
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install transformers torch"
            )
    
    elif model_type == "paraphrase":
        try:
            from .adapters.hf_paraphrase import HuggingFaceParaphraseBridge
            return HuggingFaceParaphraseBridge(
                model_name=config.model_name or "tuner007/pegasus_paraphrase",
                config=config
            )
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install transformers torch"
            )
    
    elif model_type == "translation" or model_type == "translate":
        try:
            from .adapters.hf_translation import HuggingFaceTranslationBridge
            return HuggingFaceTranslationBridge(
                model_name=config.model_name or "Helsinki-NLP/opus-mt-en-fr",
                config=config
            )
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install transformers torch"
            )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def process_text(bridge: BridgeBase, text: str) -> Dict[str, any]:
    """
    Process a single text with the bridge adapter.
    
    Args:
        bridge: Configured bridge adapter
        text: Text to process
        
    Returns:
        JSON-serializable result dictionary
    """
    result = bridge.from_text(text)
    return result.to_json()


def process_stream(bridge: BridgeBase, input_stream: TextIO, 
                  output_stream: TextIO, batch_size: int = 1,
                  parallel: bool = False, max_workers: int = 4,
                  question: Optional[str] = None,
                  show_progress: bool = False) -> int:
    """
    Process a stream of text with the bridge adapter.
    
    Args:
        bridge: Configured bridge adapter
        input_stream: Input text stream
        output_stream: Output JSON stream
        batch_size: Number of lines to process at once
        parallel: Whether to process batches in parallel
        max_workers: Maximum number of worker processes for parallel processing
        question: Optional question for QA models
        show_progress: Whether to show a progress bar
    """
    # Set question for QA models if provided
    if question and hasattr(bridge, 'set_question'):
        bridge.set_question(question)
    
    # Count lines for progress bar if needed
    total_lines = None
    if show_progress and hasattr(input_stream, 'seekable') and input_stream.seekable():
        try:
            # Count lines efficiently
            current_pos = input_stream.tell()
            total_lines = sum(1 for _ in input_stream if _.strip())
            input_stream.seek(current_pos)  # Reset position
        except (IOError, AttributeError, UnsupportedOperation):
            # If we can't count lines, we'll use an indeterminate progress bar
            pass
    
    # Read lines in chunks to avoid loading entire file into memory
    def read_chunks(stream, chunk_size):
        lines = []
        for line in stream:
            line_stripped = line.strip()
            if line_stripped:
                lines.append(line_stripped)
                if len(lines) >= chunk_size:
                    yield lines
                    lines = []
        if lines:
            yield lines
    
    # Create progress bar if requested
    pbar = None
    if show_progress:
        pbar = tqdm(total=total_lines, desc="Processing", unit="texts")
    
    start_time = time.time()
    processed_count = 0
    
    try:
        # Process in batches
        for batch in read_chunks(input_stream, batch_size):
            
            if parallel and batch_size > 1:
                try:
                    import concurrent.futures
                    # Use ThreadPoolExecutor instead of ProcessPoolExecutor for shared memory models
                    # This avoids serialization issues with complex models
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Process texts in parallel
                        future_to_text = {executor.submit(process_text, bridge, text): text for text in batch}
                        for future in concurrent.futures.as_completed(future_to_text):
                            try:
                                result = future.result()
                                output_stream.write(json.dumps(result) + "\n")
                                output_stream.flush()
                                processed_count += 1
                                if pbar:
                                    pbar.update(1)
                            except Exception as e:
                                print(f"Error processing text: {e}", file=sys.stderr)
                except ImportError:
                    # Fall back to sequential processing if concurrent.futures is not available
                    for text in batch:
                        try:
                            result = process_text(bridge, text)
                            output_stream.write(json.dumps(result) + "\n")
                            output_stream.flush()
                            processed_count += 1
                            if pbar:
                                pbar.update(1)
                        except Exception as e:
                            print(f"Error processing text: {e}", file=sys.stderr)
            else:
                # Sequential processing
                for text in batch:
                    try:
                        result = process_text(bridge, text)
                        output_stream.write(json.dumps(result) + "\n")
                        output_stream.flush()
                        processed_count += 1
                        if pbar:
                            pbar.update(1)
                    except Exception as e:
                        print(f"Error processing text: {e}", file=sys.stderr)
    finally:
        if pbar:
            pbar.close()
    
    # Print summary statistics
    elapsed_time = time.time() - start_time
    if processed_count > 0 and show_progress:
        print(f"Processed {processed_count} texts in {elapsed_time:.4f}s "
              f"({processed_count / elapsed_time:.2f} texts/sec)" if elapsed_time > 0 else
              f"Processed {processed_count} texts in {elapsed_time:.4f}s", 
              file=sys.stderr)
        
        # Print metrics if available
        if hasattr(bridge, 'get_metrics'):
            metrics = bridge.get_metrics()
            if metrics:
                print("Performance metrics:", file=sys.stderr)
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}", file=sys.stderr)
                    else:
                        print(f"  {key}: {value}", file=sys.stderr)
                        
    return processed_count


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="BridgeNLP: Universal NLP model integration"
    )
    
    # Check if spaCy is installed
    try:
        import spacy
    except ImportError:
        print("Warning: spaCy is not installed. Some features may not work.", file=sys.stderr)
        print("Install spaCy with: pip install spacy", file=sys.stderr)
    
    # Main command
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction on text")
    predict_parser.add_argument(
        "--model", required=True,
        help="Model type to use (coref, srl, ner, sentiment, classify, qa)"
    )
    predict_parser.add_argument(
        "--model-name", 
        help="Specific model name or path"
    )
    
    # Input options
    input_group = predict_parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--text", 
        help="Text to process"
    )
    input_group.add_argument(
        "--file", 
        help="File containing text to process"
    )
    
    # Output options
    predict_parser.add_argument(
        "--output", 
        help="Output file (default: stdout)"
    )
    predict_parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for processing (default: 1)"
    )
    predict_parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON output"
    )
    
    # QA-specific options
    predict_parser.add_argument(
        "--question",
        help="Question for question-answering models"
    )
    
    # Batch processing options
    predict_parser.add_argument(
        "--parallel", action="store_true",
        help="Process batches in parallel (when possible)"
    )
    predict_parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum number of worker processes for parallel processing"
    )
    
    # Configuration options
    predict_parser.add_argument(
        "--config", 
        help="Path to JSON configuration file"
    )
    predict_parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use ('cpu' for CPU, number for specific GPU, or 'cuda')"
    )
    predict_parser.add_argument(
        "--collect-metrics", action="store_true",
        help="Collect and report performance metrics"
    )
    predict_parser.add_argument(
        "--progress", action="store_true",
        help="Show progress bar during processing"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "predict":
        try:
            # Load configuration if provided
            config = None
            if args.config:
                config = BridgeConfig.from_json(args.config)
            else:
                # Create config from command line arguments
                config = BridgeConfig(
                    model_type=args.model,
                    model_name=args.model_name,
                    device=args.device,
                    batch_size=args.batch_size,
                    use_threading=args.parallel,
                    num_threads=args.max_workers,
                    collect_metrics=args.collect_metrics
                )
            
            # Load bridge with configuration
            bridge = load_bridge(args.model, args.model_name, config)
            
            # Set question for QA models if provided
            if args.question and hasattr(bridge, 'set_question'):
                bridge.set_question(args.question)
            
            # Use context manager for proper resource cleanup
            with bridge:
                # Determine input source
                if args.text:
                    result = process_text(bridge, args.text)
                    
                    # Determine output destination
                    if args.output:
                        with open(args.output, "w") as f:
                            if args.pretty:
                                json.dump(result, f, indent=2)
                            else:
                                json.dump(result, f)
                    else:
                        if args.pretty:
                            json.dump(result, sys.stdout, indent=2)
                        else:
                            json.dump(result, sys.stdout)
                        sys.stdout.write("\n")
                
                elif args.file:
                    # Process file
                    try:
                        with open(args.file, "r") as input_file:
                            if args.output:
                                with open(args.output, "w") as output_file:
                                    process_stream(
                                        bridge, input_file, output_file, 
                                        batch_size=args.batch_size,
                                        parallel=args.parallel,
                                        max_workers=args.max_workers,
                                        question=args.question,
                                        show_progress=args.progress
                                    )
                            else:
                                process_stream(
                                    bridge, input_file, sys.stdout,
                                    batch_size=args.batch_size,
                                    parallel=args.parallel,
                                    max_workers=args.max_workers,
                                    question=args.question,
                                    show_progress=args.progress
                                )
                    except FileNotFoundError:
                        print(f"Error: Input file not found: {args.file}", file=sys.stderr)
                        sys.exit(1)
                
                else:
                    # Process stdin to stdout
                    process_stream(
                        bridge, sys.stdin, sys.stdout,
                        batch_size=args.batch_size,
                        parallel=args.parallel,
                        max_workers=args.max_workers,
                        question=args.question,
                        show_progress=args.progress
                    )
        
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            
            # Provide helpful installation instructions based on the model type
            if args.model in ["coref", "spanbert-coref"]:
                print("\nTo install AllenNLP dependencies, run:", file=sys.stderr)
                print("pip install allennlp allennlp-models", file=sys.stderr)
            elif args.model in ["srl", "sentiment", "classify", "classification", "qa", "question-answering", "embeddings", "embedding"]:
                print("\nTo install Hugging Face dependencies, run:", file=sys.stderr)
                print("pip install transformers torch", file=sys.stderr)
            elif args.model in ["ner", "spacy-ner"]:
                print("\nTo install spaCy and download models, run:", file=sys.stderr)
                print(f"pip install spacy", file=sys.stderr)
                print(f"python -m spacy download {args.model_name or 'en_core_web_sm'}", file=sys.stderr)
            
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command is None:
        parser.print_help()
        sys.exit(1)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
