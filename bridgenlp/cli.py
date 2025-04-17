#!/usr/bin/env python
"""
Command-line interface for BridgeNLP.

This module provides a command-line tool for using BridgeNLP adapters
without writing Python code.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, TextIO, Union

import spacy

from .base import BridgeBase
from .result import BridgeResult


def load_bridge(model_type: str, model_name: Optional[str] = None) -> BridgeBase:
    """
    Load a bridge adapter based on model type and name.
    
    Args:
        model_type: Type of model to load (e.g., 'coref', 'srl', 'ner', 'sentiment', 'classify', 'qa')
        model_name: Optional specific model name
        
    Returns:
        Configured bridge adapter
        
    Raises:
        ImportError: If required dependencies are not installed
        ValueError: If model type is not recognized
    """
    if model_type == "coref" or model_type == "spanbert-coref":
        try:
            from .adapters.allen_coref import AllenNLPCorefBridge
            return AllenNLPCorefBridge(
                model_name=model_name or "coref-spanbert"
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
                model_name=model_name or "Davlan/bert-base-multilingual-cased-srl-nli"
            )
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install bridgenlp[huggingface]"
            )
    
    elif model_type == "ner" or model_type == "spacy-ner":
        try:
            from .adapters.spacy_ner import SpacyNERBridge
            return SpacyNERBridge(
                model_name=model_name or "en_core_web_sm"
            )
        except ImportError as e:
            raise ImportError(
                f"Error loading spaCy NER model: {str(e)}. "
                f"Make sure the model is installed with: "
                f"python -m spacy download {model_name or 'en_core_web_sm'}"
            )
    
    elif model_type == "sentiment":
        try:
            from .adapters.hf_sentiment import HuggingFaceSentimentBridge
            return HuggingFaceSentimentBridge(
                model_name=model_name or "distilbert-base-uncased-finetuned-sst-2-english"
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
                model_name=model_name or "facebook/bart-large-mnli"
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
                model_name=model_name or "deepset/roberta-base-squad2"
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
            return NLTKBridge()
        except ImportError:
            raise ImportError(
                "NLTK not found. Install with: pip install nltk"
            )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def process_text(bridge: BridgeBase, text: str) -> Dict:
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
                  question: Optional[str] = None) -> None:
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
    """
    # Set question for QA models if provided
    if question and hasattr(bridge, 'set_question'):
        bridge.set_question(question)
    
    # Read lines in chunks to avoid loading entire file into memory
    def read_chunks(stream, chunk_size):
        lines = []
        for line in stream:
            if line.strip():
                lines.append(line.strip())
                if len(lines) >= chunk_size:
                    yield lines
                    lines = []
        if lines:
            yield lines
    
    # Process in batches
    for batch in read_chunks(input_stream, batch_size):
        
        if parallel and batch_size > 1:
            try:
                import concurrent.futures
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Process texts in parallel
                    future_to_text = {executor.submit(process_text, bridge, text): text for text in batch}
                    for future in concurrent.futures.as_completed(future_to_text):
                        try:
                            result = future.result()
                            output_stream.write(json.dumps(result) + "\n")
                            output_stream.flush()
                        except Exception as e:
                            print(f"Error processing text: {e}", file=sys.stderr)
            except ImportError:
                # Fall back to sequential processing if concurrent.futures is not available
                for text in batch:
                    result = process_text(bridge, text)
                    output_stream.write(json.dumps(result) + "\n")
                    output_stream.flush()
        else:
            # Sequential processing
            for text in batch:
                result = process_text(bridge, text)
                output_stream.write(json.dumps(result) + "\n")
                output_stream.flush()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="BridgeNLP: Universal NLP model integration"
    )
    
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "predict":
        try:
            bridge = load_bridge(args.model, args.model_name)
            
            # Set question for QA models if provided
            if args.question and hasattr(bridge, 'set_question'):
                bridge.set_question(args.question)
            
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
                with open(args.file, "r") as input_file:
                    if args.output:
                        with open(args.output, "w") as output_file:
                            process_stream(
                                bridge, input_file, output_file, 
                                batch_size=args.batch_size,
                                parallel=args.parallel,
                                max_workers=args.max_workers,
                                question=args.question
                            )
                    else:
                        process_stream(
                            bridge, input_file, sys.stdout,
                            batch_size=args.batch_size,
                            parallel=args.parallel,
                            max_workers=args.max_workers,
                            question=args.question
                        )
            
            else:
                # Process stdin to stdout
                process_stream(
                    bridge, sys.stdin, sys.stdout,
                    batch_size=args.batch_size,
                    parallel=args.parallel,
                    max_workers=args.max_workers,
                    question=args.question
                )
        
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
