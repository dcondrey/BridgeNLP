"""
Base abstract classes for BridgeNLP adapters.
"""

from abc import ABC, abstractmethod
import contextlib
import threading
import time
from typing import Dict, List, Optional, Union, Any

try:
    import spacy
    from spacy.tokens import Doc
except ImportError:
    # Provide a helpful error message but allow the module to be imported
    print("Warning: spaCy not installed. Install with: pip install spacy")
    spacy = None
    Doc = Any

from .config import BridgeConfig
from .result import BridgeResult


class BridgeBase(ABC):
    """
    Abstract base class for all bridge adapters.
    
    All bridge adapters must implement these methods to ensure
    consistent behavior across different model integrations.
    """
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        """
        Initialize the bridge adapter with optional configuration.
        
        Args:
            config: Configuration for the adapter
        """
        self.config = config
        self._metrics = {
            "num_calls": 0,
            "total_time": 0.0,
            "total_tokens": 0,
            "errors": 0
        }
        self._metrics_lock = threading.RLock()
    
    @abstractmethod
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text and return structured results.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing the processed information
        """
        # The context manager should be used in implementations, not in the abstract method
        pass
    
    @abstractmethod
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text and return structured results.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing the processed information
        """
        # The context manager should be used in implementations, not in the abstract method
        pass
    
    def from_batch(self, texts: List[str]) -> List[BridgeResult]:
        """
        Process a batch of texts for efficient processing.
        
        This method can be overridden by adapters that support batch processing.
        The default implementation calls from_text for each item.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of BridgeResult objects
        """
        return [self.from_text(text) for text in texts]
    
    def from_token_batch(self, token_lists: List[List[str]]) -> List[BridgeResult]:
        """
        Process a batch of token lists for efficient processing.
        
        This method can be overridden by adapters that support batch processing.
        The default implementation calls from_tokens for each item.
        
        Args:
            token_lists: List of token lists to process
            
        Returns:
            List of BridgeResult objects
        """
        return [self.from_tokens(tokens) for tokens in token_lists]
    
    @abstractmethod
    def from_spacy(self, doc: "Doc") -> "Doc":
        """
        Process a spaCy Doc and return an enhanced Doc with results attached.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            The same Doc with additional attributes attached
        """
        # The context manager should be used in implementations, not in the abstract method
        pass
    
    def from_spacy_batch(self, docs: List["Doc"]) -> List["Doc"]:
        """
        Process a batch of spaCy Docs for efficient processing.
        
        This method can be overridden by adapters that support batch processing.
        The default implementation calls from_spacy for each item.
        
        Args:
            docs: List of spaCy Docs to process
            
        Returns:
            List of processed spaCy Docs
        """
        return [self.from_spacy(doc) for doc in docs]
    
    @contextlib.contextmanager
    def _measure_performance(self):
        """
        Context manager to measure performance metrics.
        
        This automatically tracks call count, processing time, and errors
        for all processing methods.
        """
        if not hasattr(self, "config") or not self.config or not self.config.collect_metrics:
            yield
            return
        
        start_time = time.time()
        
        # Thread-safe increment of call count
        with self._metrics_lock:
            self._metrics["num_calls"] += 1
        
        try:
            yield
        except Exception as e:
            with self._metrics_lock:
                self._metrics["errors"] += 1
            raise e
        finally:
            elapsed = time.time() - start_time
            with self._metrics_lock:
                self._metrics["total_time"] += elapsed
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for this adapter.
        
        Returns:
            Dictionary of metrics including average processing time
        """
        with self._metrics_lock:
            metrics = dict(self._metrics)
        
        # Calculate derived metrics
        if metrics["num_calls"] > 0:
            metrics["avg_time"] = metrics["total_time"] / metrics["num_calls"]
            if metrics["total_tokens"] > 0 and metrics["total_time"] > 0.00001:  # Avoid division by very small numbers
                metrics["tokens_per_second"] = metrics["total_tokens"] / metrics["total_time"]
            elif metrics["total_tokens"] > 0:
                metrics["tokens_per_second"] = float('inf')  # Indicate extremely fast processing
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self._metrics_lock:
            self._metrics = {
                "num_calls": 0,
                "total_time": 0.0,
                "total_tokens": 0,
                "errors": 0
            }
    
    def __enter__(self):
        """Support for context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when used as a context manager."""
        self.cleanup()
        return False  # Don't suppress exceptions
    
    def cleanup(self):
        """
        Clean up resources used by this adapter.
        
        Override in subclasses to implement specific cleanup logic.
        """
        pass
