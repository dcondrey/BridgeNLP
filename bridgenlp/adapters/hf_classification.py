"""
Hugging Face text classification adapter for BridgeNLP.
"""

import functools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.tokens import Doc

from ..aligner import TokenAligner
from ..base import BridgeBase
from ..result import BridgeResult


class HuggingFaceClassificationBridge(BridgeBase):
    """
    Bridge adapter for Hugging Face's text classification models.
    
    This adapter integrates transformer-based text classification models from Hugging Face
    with token-based pipelines like spaCy.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli", 
                 device: int = -1, labels: Optional[List[str]] = None):
        """
        Initialize the text classification bridge.
        
        Args:
            model_name: Name or path of the Hugging Face model to use
            device: Device to run the model on (-1 for CPU, 0+ for GPU)
            labels: Optional list of labels for zero-shot classification
                   (default: ["positive", "negative", "neutral"])
        
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
        self.labels = labels or ["positive", "negative", "neutral"]
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
                "zero-shot-classification", 
                model=self.model_name,
                device=device
            )
        return self._pipeline
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text with text classification.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing text classification results
        """
        if not text.strip():
            return BridgeResult(tokens=[])
        
        # Run the model
        results = self.pipeline(text, self.labels, multi_label=True)
        
        # Extract tokens (we'll use a simple whitespace tokenizer for now)
        tokens = text.split()
        
        # Process the classification results
        roles = []
        for label, score in zip(results["labels"], results["scores"]):
            role = {
                "role": "CLASS",
                "label": label,
                "score": score,
                "text": text
            }
            roles.append(role)
        
        return BridgeResult(
            tokens=tokens,
            roles=roles
        )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text with text classification.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing text classification results
        """
        text = " ".join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and attach text classification results.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The same Doc with text classification results attached
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
        # Clear the cached pipeline to free memory
        if hasattr(self, '_pipeline') and self._pipeline is not None:
            # Clear any GPU memory if applicable
            try:
                import torch
                if torch.cuda.is_available() and self.device >= 0:
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Remove references to large objects
            if hasattr(self._pipeline, 'model'):
                self._pipeline.model = None
            if hasattr(self._pipeline, 'tokenizer'):
                self._pipeline.tokenizer = None
            
            self._pipeline = None
