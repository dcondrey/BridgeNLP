"""
Hugging Face semantic role labeling adapter for BridgeNLP.
"""

import functools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.tokens import Doc

from ..aligner import TokenAligner
from ..base import BridgeBase
from ..result import BridgeResult


class HuggingFaceSRLBridge(BridgeBase):
    """
    Bridge adapter for Hugging Face's semantic role labeling models.
    
    This adapter integrates transformer-based SRL models from Hugging Face
    with token-based pipelines like spaCy.
    """
    
    def __init__(self, model_name: str = "Davlan/bert-base-multilingual-cased-srl-nli", 
                 device: int = -1):
        """
        Initialize the semantic role labeling bridge.
        
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
                "token-classification", 
                model=self.model_name,
                device=device,
                aggregation_strategy="simple"
            )
        return self._pipeline
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text with semantic role labeling.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing semantic roles
        """
        if not text.strip():
            return BridgeResult(tokens=[])
        
        # Run the model
        results = self.pipeline(text)
        
        # Extract tokens (we'll use a simple whitespace tokenizer for now)
        tokens = text.split()
        
        # Process the roles
        roles = []
        for entity in results:
            role = {
                "role": entity["entity_group"],
                "text": entity["word"],
                "score": entity["score"],
                "start": entity["start"],
                "end": entity["end"]
            }
            roles.append(role)
        
        return BridgeResult(
            tokens=tokens,
            roles=roles
        )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text with semantic role labeling.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing semantic roles
        """
        text = " ".join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and attach semantic roles.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The same Doc with semantic roles attached
        """
        # Get raw text from the document
        text = doc.text
        
        # Process with the model
        result = self.from_text(text)
        
        # Align roles to spaCy token boundaries
        aligned_roles = []
        for role in result.roles:
            # Align to spaCy tokens
            span = self.aligner.align_char_span(doc, role["start"], role["end"])
            if span is not None:
                aligned_role = role.copy()
                aligned_role["start_token"] = span.start
                aligned_role["end_token"] = span.end
                aligned_role["tokens"] = [t.text for t in span]
                aligned_roles.append(aligned_role)
        
        # Create a new result with aligned roles
        aligned_result = BridgeResult(
            tokens=[t.text for t in doc],
            roles=aligned_roles
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
