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
from ..result import BridgeResult


class HuggingFaceSentimentBridge(BridgeBase):
    """
    Bridge adapter for Hugging Face's sentiment analysis models.
    
    This adapter integrates transformer-based sentiment analysis models from Hugging Face
    with token-based pipelines like spaCy.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english", 
                 device: int = -1):
        """
        Initialize the sentiment analysis bridge.
        
        Args:
            model_name: Name or path of the Hugging Face model to use
            device: Device to run the model on (-1 for CPU, 0+ for GPU)
        
        Raises:
            ImportError: If Hugging Face dependencies are not installed
        """
        try:
            import torch
            import transformers
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "Hugging Face dependencies not found. Install with: "
                "pip install bridgenlp[huggingface]"
            )
        
        self.model_name = model_name
        self.device = device
        self.aligner = TokenAligner()
        self._pipeline = None
    
    @property
    @functools.lru_cache(maxsize=1)
    def pipeline(self):
        """
        Lazy-load the Hugging Face pipeline.
        
        Returns:
            Loaded Hugging Face pipeline
        """
        from transformers import pipeline
        import torch
        
        if self._pipeline is None:
            device = self.device if self.device >= 0 else "cpu"
            self._pipeline = pipeline(
                "sentiment-analysis", 
                model=self.model_name,
                device=device
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
        text = " ".join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and attach sentiment analysis results.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The same Doc with sentiment analysis results attached
        """
        # Get raw text from the document
        text = doc.text
        
        # Process with the model
        result = self.from_text(text)
        
        # Create a new result with the same roles
        aligned_result = BridgeResult(
            tokens=[t.text for t in doc],
            roles=result.roles
        )
        
        # Attach to the document
        return aligned_result.attach_to_spacy(doc)
    
    def __del__(self):
        """
        Clean up resources when the object is deleted.
        
        This method ensures that the model is properly unloaded
        to prevent memory leaks.
        """
        try:
            # Clear the cached pipeline to free memory
            if hasattr(self, '_pipeline') and self._pipeline is not None:
                # Clear any GPU memory if applicable
                try:
                    import torch
                    if torch.cuda.is_available() and self.device >= 0:
                        torch.cuda.empty_cache()
                except (ImportError, RuntimeError, AttributeError):
                    pass
                
                # Remove references to large objects
                if hasattr(self._pipeline, 'model'):
                    self._pipeline.model = None
                if hasattr(self._pipeline, 'tokenizer'):
                    self._pipeline.tokenizer = None
                
                self._pipeline = None
        except Exception:
            # Ensure no exceptions are raised during garbage collection
            pass
