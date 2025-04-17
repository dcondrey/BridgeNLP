"""
Base abstract classes for BridgeNLP adapters.
"""

from abc import ABC, abstractmethod
from typing import List

import spacy

from .result import BridgeResult


class BridgeBase(ABC):
    """
    Abstract base class for all bridge adapters.
    
    All bridge adapters must implement these methods to ensure
    consistent behavior across different model integrations.
    """
    
    @abstractmethod
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text and return structured results.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing the processed information
        """
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
        pass
