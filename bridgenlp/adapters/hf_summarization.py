"""
Hugging Face text summarization adapter for BridgeNLP.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    pipeline = None

from ..base import BridgeBase
from ..config import BridgeConfig
from ..result import BridgeResult
from ..utils import (
    configure_device,
    get_param_with_fallback,
    get_or_create_model,
    unload_model,
    validate_text_input,
    get_model_memory_usage,
    create_model_key
)


class HuggingFaceSummarizationBridge(BridgeBase):
    """
    Bridge adapter for Hugging Face summarization models.
    
    This adapter integrates text summarization models from the Hugging Face
    Transformers library with the BridgeNLP framework.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-cnn",
                 max_length: int = 130,
                 min_length: int = 30,
                 do_sample: bool = False,
                 truncation: bool = True,
                 batch_size: int = 1,
                 lazy_loading: bool = False,
                 config: Optional[BridgeConfig] = None):
        """
        Initialize the summarization bridge.
        
        Args:
            model_name: Name or path of the Hugging Face model to use
            max_length: Maximum length of generated summaries
            min_length: Minimum length of generated summaries
            do_sample: Whether to use sampling for generation
            truncation: Whether to truncate input sequences longer than model max length
            batch_size: Batch size for batch processing (default: 1)
            lazy_loading: Whether to load the model only when first used (default: False)
            config: Optional configuration for the adapter
            
        Raises:
            ImportError: If required dependencies are not installed
        """
        # Check dependencies
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise ImportError(
                "Hugging Face Transformers not installed. "
                "Install with: pip install transformers torch"
            )
        
        # Always call the parent constructor first
        super().__init__(config)
        
        # Store model name and parameters using utility functions
        self.model_name = get_param_with_fallback(
            model_name, config, "model_name", default_value="facebook/bart-large-cnn"
        )
        
        self.max_length = get_param_with_fallback(
            max_length, config, "max_length", default_value=130
        )
        
        self.min_length = get_param_with_fallback(
            min_length, config, "params", "min_length", default_value=30
        )
        
        self.do_sample = get_param_with_fallback(
            do_sample, config, "params", "do_sample", default_value=False
        )
        
        self.truncation = get_param_with_fallback(
            truncation, config, "params", "truncation", default_value=True
        )
        
        self.batch_size = get_param_with_fallback(
            batch_size, config, "batch_size", default_value=1
        )
        
        self.lazy_loading = get_param_with_fallback(
            lazy_loading, config, "params", "lazy_loading", default_value=False
        )
        
        # Configure device using utility function
        self.device = configure_device(
            get_param_with_fallback(None, config, "device", default_value=-1)
        )
        
        # Create a unique key for this model in the registry
        self.model_key = create_model_key(self.model_name, "summarization", self.device)
        
        # Initialize model and tokenizer if not using lazy loading
        self.summarizer = None
        if not self.lazy_loading:
            self._init_model()
        
        # Thread lock for model inference
        self._model_lock = threading.RLock()
        
        # Track memory usage
        self.memory_usage = 0.0
        
        # Performance metrics
        self._metrics.update({
            "total_characters": 0,
            "batch_calls": 0,
            "model_load_time": 0.0,
        })
    
    def _init_model(self):
        """Initialize the model and tokenizer."""
        if self.summarizer is not None:
            return
            
        start_time = time.time()
        
        try:
            # Use global model registry to share models between adapters
            def create_summarizer():
                return pipeline(
                    "summarization", 
                    model=self.model_name, 
                    tokenizer=self.model_name,
                    device=self.device,
                    framework="pt"  # Use PyTorch
                )
            
            # Get or create the model
            self.summarizer = get_or_create_model(
                self.model_key,
                create_summarizer
            )
            
            # Track memory usage if using PyTorch model
            if hasattr(self.summarizer, "model"):
                self.memory_usage = get_model_memory_usage(self.summarizer.model)
                
        except Exception as e:
            raise ImportError(f"Error loading model {self.model_name}: {str(e)}")
        finally:
            # Record model loading time
            with self._metrics_lock:
                self._metrics["model_load_time"] += time.time() - start_time
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Generate a summary from raw text.
        
        Args:
            text: Raw text to summarize
            
        Returns:
            BridgeResult containing the summary
        
        Raises:
            ValueError: If the input text is invalid
            RuntimeError: If summarization fails
        """
        with self._measure_performance():
            # Validate and clean input
            try:
                text = validate_text_input(text)
                if not text:
                    return BridgeResult(tokens=[])
            except ValueError as e:
                raise ValueError(f"Invalid input text: {str(e)}")
            
            # Lazy load model if needed
            if self.summarizer is None:
                self._init_model()
            
            # Generate summary with the model
            with self._model_lock:
                try:
                    summary = self.summarizer(
                        text,
                        max_length=self.max_length,
                        min_length=self.min_length,
                        do_sample=self.do_sample,
                        truncation=self.truncation
                    )
                except Exception as e:
                    # Handle inference errors gracefully
                    raise RuntimeError(f"Summarization failed: {str(e)}")
            
            # Process the summary result
            if isinstance(summary, list) and len(summary) > 0:
                summary_text = summary[0].get('summary_text', '')
            else:
                summary_text = ''
            
            # Perform better tokenization for the summary
            tokens = self._tokenize_text(summary_text)
            
            # Original text analysis with better tokenization
            original_tokens = self._tokenize_text(text)
            
            # Store the summarization result as a role with more metadata
            roles = [{
                "role": "SUMMARY",
                "text": summary_text,
                "original_length": len(original_tokens),
                "summary_length": len(tokens),
                "compression_ratio": len(tokens) / len(original_tokens) if len(original_tokens) > 0 else 0,
                "model": self.model_name,
                "generation_params": {
                    "max_length": self.max_length,
                    "min_length": self.min_length,
                    "do_sample": self.do_sample
                }
            }]
            
            # Update metrics
            with self._metrics_lock:
                self._metrics["total_tokens"] += len(tokens)
                self._metrics["total_characters"] += len(text)
            
            return BridgeResult(
                tokens=tokens,
                roles=roles
            )
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Better tokenization that handles more languages and special cases.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # If we have the tokenizer available, use it for better tokenization
        if hasattr(self.summarizer, "tokenizer"):
            try:
                # Use the model's tokenizer but convert back to strings
                token_ids = self.summarizer.tokenizer.encode(text)
                return [self.summarizer.tokenizer.decode([token_id]) for token_id in token_ids]
            except Exception:
                pass
                
        # Fall back to simple whitespace tokenization
        return text.split()
        
    def from_batch(self, texts: List[str]) -> List[BridgeResult]:
        """
        Process a batch of texts for efficient processing.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of BridgeResult objects
        """
        with self._measure_performance():
            # Update batch calls metrics
            with self._metrics_lock:
                self._metrics["batch_calls"] += 1
            
            # Validate inputs
            valid_texts = []
            for text in texts:
                try:
                    cleaned = validate_text_input(text)
                    valid_texts.append(cleaned if cleaned else "")
                except ValueError:
                    valid_texts.append("")
            
            # Skip processing if all texts are empty
            if not any(valid_texts):
                return [BridgeResult(tokens=[]) for _ in texts]
                
            # Lazy load model if needed
            if self.summarizer is None:
                self._init_model()
                
            # Process texts in batches
            all_summaries = []
            
            with self._model_lock:
                try:
                    # Use batched processing if available in the pipeline
                    if hasattr(self.summarizer, "model") and self.batch_size > 1:
                        # Process in chunks of batch_size
                        for i in range(0, len(valid_texts), self.batch_size):
                            batch = valid_texts[i:i+self.batch_size]
                            batch_summaries = self.summarizer(
                                batch,
                                max_length=self.max_length,
                                min_length=self.min_length,
                                do_sample=self.do_sample,
                                truncation=self.truncation
                            )
                            all_summaries.extend(batch_summaries)
                    else:
                        # Fall back to individual processing
                        for text in valid_texts:
                            if not text:
                                all_summaries.append({"summary_text": ""})
                                continue
                                
                            summary = self.summarizer(
                                text,
                                max_length=self.max_length,
                                min_length=self.min_length,
                                do_sample=self.do_sample,
                                truncation=self.truncation
                            )
                            
                            if isinstance(summary, list) and len(summary) > 0:
                                all_summaries.append(summary[0])
                            else:
                                all_summaries.append({"summary_text": ""})
                except Exception as e:
                    # Handle batch processing errors
                    raise RuntimeError(f"Batch summarization failed: {str(e)}")
            
            # Create BridgeResult objects for each summary
            results = []
            for i, (text, summary_obj) in enumerate(zip(valid_texts, all_summaries)):
                summary_text = summary_obj.get("summary_text", "")
                
                # Tokenize
                tokens = self._tokenize_text(summary_text)
                original_tokens = self._tokenize_text(text)
                
                # Create role information
                roles = [{
                    "role": "SUMMARY",
                    "text": summary_text,
                    "original_length": len(original_tokens),
                    "summary_length": len(tokens),
                    "compression_ratio": len(tokens) / len(original_tokens) if len(original_tokens) > 0 else 0,
                    "model": self.model_name,
                    "batch_index": i
                }]
                
                # Update metrics
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(tokens)
                    self._metrics["total_characters"] += len(text)
                
                results.append(BridgeResult(tokens=tokens, roles=roles))
            
            return results
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Generate a summary from pre-tokenized text.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing the summary
        """
        if not tokens:
            return BridgeResult(tokens=[])
            
        # Convert tokens to text and call from_text
        text = ' '.join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc) -> "spacy.tokens.Doc":
        """
        Generate a summary from a spaCy Doc.
        
        Args:
            doc: spaCy Doc to summarize
            
        Returns:
            The same Doc with summary information attached
        """
        if doc is None:
            raise ValueError("Input Doc cannot be None")
            
        with self._measure_performance():
            # Get the text from the Doc
            text = doc.text
            
            # Generate summary
            result = self.from_text(text)
            
            # Attach the result to the Doc
            return result.attach_to_spacy(doc)
    
    def get_metrics(self) -> Dict[str, Union[int, float, str]]:
        """
        Get enhanced performance metrics for this adapter.
        
        Returns:
            Dictionary of metrics with additional information
        """
        # Get base metrics
        metrics = super().get_metrics()
        
        # Add memory usage information
        metrics["memory_usage_mb"] = self.memory_usage
        
        # Add efficiency metrics
        if metrics["total_tokens"] > 0 and metrics["total_characters"] > 0:
            metrics["characters_per_token"] = metrics["total_characters"] / metrics["total_tokens"]
        
        if metrics["batch_calls"] > 0:
            metrics["avg_batch_size"] = metrics["num_calls"] / metrics["batch_calls"]
            
        if metrics["model_load_time"] > 0:
            metrics["load_time_seconds"] = metrics["model_load_time"]
            
        # Add model information
        metrics["model_name"] = self.model_name
        metrics["device"] = f"GPU:{self.device}" if self.device >= 0 else "CPU"
        metrics["lazy_loading"] = self.lazy_loading
        
        return metrics
    
    def cleanup(self):
        """
        Clean up resources used by this adapter.
        
        This method is called when the adapter is used as a context manager
        or when it's garbage collected.
        """
        # Unload the model if requested in config
        unload = (hasattr(self, "config") and self.config and 
                  hasattr(self.config, "unload_on_del") and self.config.unload_on_del)
                  
        if unload:
            with self._model_lock:
                # Unregister from global registry instead of just local cleanup
                unload_model(self.model_key)
                self.summarizer = None
    
    def __repr__(self) -> str:
        """
        String representation of this adapter.
        
        Returns:
            A string with adapter information
        """
        status = "loaded" if self.summarizer is not None else "not loaded"
        device = f"GPU:{self.device}" if self.device >= 0 else "CPU"
        return f"HuggingFaceSummarizationBridge(model='{self.model_name}', device={device}, status={status})"