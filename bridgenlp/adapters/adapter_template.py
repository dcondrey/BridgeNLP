"""
Template for creating new bridge adapters.

This file serves as a reference implementation for creating new bridge adapters
that properly integrate with the configuration system and resource management.
"""

from typing import List, Optional

import spacy
from spacy.tokens import Doc

from ..base import BridgeBase
from ..config import BridgeConfig
from ..result import BridgeResult


class TemplateBridge(BridgeBase):
    """
    Template bridge adapter for reference.
    
    This class demonstrates the proper implementation of a bridge adapter
    with configuration support and resource management.
    """
    
    def __init__(self, model_name: Optional[str] = None, config: Optional[BridgeConfig] = None):
        """
        Initialize the bridge adapter.
        
        Args:
            model_name: Name or path of the model to load
            config: Configuration for the adapter
        """
        # Always call the parent constructor first
        super().__init__(config)
        
        # Store model name, using config if provided
        self.model_name = model_name or (config.model_name if config else None) or "default-model"
        
        # Extract configuration options with defaults
        self.device = config.device if config else -1
        self.max_length = config.max_length if config else None
        self.cache_size = config.cache_size if config and config.cache_size else 1
        
        # Initialize model lazily (don't load until needed)
        self._model = None
    
    @property
    def model(self):
        """
        Lazily load the model when first accessed.
        
        Returns:
            The loaded model
        """
        if self._model is None:
            # Load the model here
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """
        Load the model from the specified source.
        
        Returns:
            The loaded model
        """
        # Implement model loading logic here
        # Example:
        # return SomeLibrary.load_model(self.model_name, device=self.device)
        return {}  # Placeholder
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text and return structured results.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing the processed information
        """
        with self._measure_performance():
            # Process the text with the model
            # Example:
            # result = self.model.process(text)
            
            # Update token count for metrics
            tokens = text.split()
            self._metrics["total_tokens"] += len(tokens)
            
            # Return standardized result
            return BridgeResult(
                tokens=tokens,
                # Add other result fields as needed
            )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text and return structured results.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing the processed information
        """
        with self._measure_performance():
            # Process the tokens with the model
            # Example:
            # result = self.model.process_tokens(tokens)
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(tokens)
            
            # Return standardized result
            return BridgeResult(
                tokens=tokens,
                # Add other result fields as needed
            )
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and return an enhanced Doc with results attached.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            The same Doc with additional attributes attached
        """
        with self._measure_performance():
            # Extract tokens from the doc
            tokens = [token.text for token in doc]
            
            # Process using the from_tokens method to avoid duplicating logic
            result = self.from_tokens(tokens)
            
            # Attach results to the doc
            return result.attach_to_spacy(doc)
    
    def cleanup(self):
        """
        Clean up resources used by this adapter.
        
        This method is called when the adapter is used as a context manager
        or when it's garbage collected.
        """
        # Implement resource cleanup here
        # Example:
        # if self._model is not None and hasattr(self.config, 'unload_on_del') and self.config.unload_on_del:
        #     self._model.unload()
        #     self._model = None
        pass
