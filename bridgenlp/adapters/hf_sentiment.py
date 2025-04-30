"""
Hugging Face sentiment analysis adapter for BridgeNLP.
"""

import functools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.tokens import Doc

from ..aligner import TokenAligner
from ..base import BridgeBase
from ..config import BridgeConfig
from ..result import BridgeResult


class HuggingFaceSentimentBridge(BridgeBase):
    """
    Bridge adapter for Hugging Face's sentiment analysis models.
    
    This adapter integrates transformer-based sentiment analysis models from Hugging Face
    with token-based pipelines like spaCy.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english", 
                 device: Union[int, str] = -1, config: Optional[BridgeConfig] = None):
        """
        Initialize the sentiment analysis bridge.
        
        Args:
            model_name: Name or path of the Hugging Face model to use
            device: Device to run the model on (-1 for CPU, 0+ for GPU, or "cuda"/"cpu")
            config: Configuration for the adapter
        
        Raises:
            ImportError: If Hugging Face dependencies are not installed
        """
        # Always call the parent constructor first
        super().__init__(config)
        
        try:
            import torch
            import transformers
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install bridgenlp[huggingface]"
            )
        
        # Store model name, using config if provided
        self.model_name = model_name or (config.model_name if config else None) or "distilbert-base-uncased-finetuned-sst-2-english"
        
        # Extract configuration options with defaults
        self.device = config.device if config else device
        self.max_length = config.max_length if config and config.max_length else None
        self.cache_size = config.cache_size if config and config.cache_size else 1
        
        self.aligner = TokenAligner()
        self._pipeline = None
    
    @property
    def pipeline(self):
        """
        Lazy-load the Hugging Face pipeline.
        
        Returns:
            Loaded Hugging Face pipeline
        """
        if self._pipeline is None:
            try:
                from transformers import pipeline
                import torch
                
                # Configure pipeline options
                pipeline_kwargs = {
                    "model": self.model_name,
                }
                
                # Handle device configuration
                if isinstance(self.device, str):
                    if self.device == "-1":
                        pipeline_kwargs["device"] = "cpu"
                    else:
                        pipeline_kwargs["device"] = self.device
                else:
                    pipeline_kwargs["device"] = self.device if self.device >= 0 else "cpu"
                
                # Add max_length if specified
                if self.max_length:
                    pipeline_kwargs["max_length"] = self.max_length
                
                self._pipeline = pipeline("sentiment-analysis", **pipeline_kwargs)
                
            except ImportError:
                raise ImportError(
                    "Hugging Face transformers not installed. "
                    "Install with: pip install transformers torch"
                )
        
        return self._pipeline
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text with sentiment analysis.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing sentiment analysis results
        """
        with self._measure_performance():
            if not text.strip():
                return BridgeResult(tokens=[])
            
            # Run the model
            results = self.pipeline(text)
            
            # Extract tokens (we'll use a simple whitespace tokenizer for now)
            tokens = text.split()
            
            # Process the sentiment
            sentiment = results[0]
            
            # Create a role-like structure for sentiment
            roles = [{
                "role": "SENTIMENT",
                "label": sentiment["label"],
                "score": sentiment["score"],
                "text": text
            }]
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(tokens)
            
            return BridgeResult(
                tokens=tokens,
                roles=roles
            )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text with sentiment analysis.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing sentiment analysis results
        """
        with self._measure_performance():
            text = " ".join(tokens)
            
            # Run the model
            results = self.pipeline(text)
            
            # Process the sentiment
            sentiment = results[0]
            
            # Create a role-like structure for sentiment
            roles = [{
                "role": "SENTIMENT",
                "label": sentiment["label"],
                "score": sentiment["score"],
                "text": text
            }]
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(tokens)
            
            return BridgeResult(
                tokens=tokens,
                roles=roles
            )
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and attach sentiment analysis results.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The same Doc with sentiment analysis results attached
        """
        with self._measure_performance():
            # Get raw text from the document
            text = doc.text
            
            # Process with the model
            results = self.pipeline(text)
            
            # Process the sentiment
            sentiment = results[0]
            
            # Create a role-like structure for sentiment
            roles = [{
                "role": "SENTIMENT",
                "label": sentiment["label"],
                "score": sentiment["score"],
                "text": text
            }]
            
            # Create a new result with the same roles
            aligned_result = BridgeResult(
                tokens=[t.text for t in doc],
                roles=roles
            )
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(doc)
            
            # Attach to the document
            return aligned_result.attach_to_spacy(doc)
    
    def cleanup(self):
        """
        Clean up resources used by this adapter.
        
        This method is called when the adapter is used as a context manager
        or when it's garbage collected.
        """
        # Unload the model if requested in config
        if (hasattr(self, "config") and self.config and 
            hasattr(self.config, "unload_on_del") and self.config.unload_on_del):
            try:
                # Clear the cached pipeline to free memory
                if self._pipeline is not None:
                    # Clear any GPU memory if applicable
                    try:
                        import torch
                        if torch.cuda.is_available():
                            if isinstance(self.device, int) and self.device >= 0:
                                torch.cuda.empty_cache()
                            elif isinstance(self.device, str) and self.device == "cuda":
                                torch.cuda.empty_cache()
                    except (ImportError, RuntimeError, AttributeError):
                        pass
                    
                    # Remove references to large objects
                    if hasattr(self._pipeline, 'model'):
                        self._pipeline.model = None
                    if hasattr(self._pipeline, 'tokenizer'):
                        self._pipeline.tokenizer = None
                    
                    self._pipeline = None
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
            except Exception:
                # Ensure no exceptions are raised during cleanup
                pass
