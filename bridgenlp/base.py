"""
Base abstract classes for BridgeNLP adapters.
"""

from abc import ABC, abstractmethod
import contextlib
import time
from typing import Dict, List, Optional

try:
    import spacy
except ImportError:
    # Provide a helpful error message but allow the module to be imported
    print("Warning: spaCy not installed. Install with: pip install spacy")
    spacy = None

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
    
    @abstractmethod
    def from_spacy(self, doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """
        Process a spaCy Doc and return an enhanced Doc with results attached.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            The same Doc with additional attributes attached
        """
        # The context manager should be used in implementations, not in the abstract method
        pass
    
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
        self._metrics["num_calls"] += 1
        
        try:
            yield
        except Exception as e:
            self._metrics["errors"] += 1
            raise e
        finally:
            self._metrics["total_time"] += time.time() - start_time
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for this adapter.
        
        Returns:
            Dictionary of metrics including average processing time
        """
        metrics = dict(self._metrics)
        
        # Calculate derived metrics
        if metrics["num_calls"] > 0:
            metrics["avg_time"] = metrics["total_time"] / metrics["num_calls"]
            if metrics["total_tokens"] > 0:
                metrics["tokens_per_second"] = metrics["total_tokens"] / metrics["total_time"]
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
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
