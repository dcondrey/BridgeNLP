"""
AllenNLP coreference resolution adapter for BridgeNLP.
"""

import functools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.tokens import Doc

from ..aligner import TokenAligner
from ..base import BridgeBase
from ..result import BridgeResult


class AllenNLPCorefBridge(BridgeBase):
    """
    Bridge adapter for AllenNLP's coreference resolution models.
    
    This adapter integrates AllenNLP's SpanBERT-based coreference resolution
    model with token-based pipelines like spaCy.
    """
    
    def __init__(self, model_name: str = "coref-spanbert", device: int = -1):
        """
        Initialize the coreference resolution bridge.
        
        Args:
            model_name: Name of the AllenNLP model to use
            device: Device to run the model on (-1 for CPU, 0+ for GPU)
        
        Raises:
            ImportError: If AllenNLP dependencies are not installed
        """
        try:
            import allennlp
            import allennlp.predictors
            from allennlp_models.coref import CorefPredictor
        except ImportError:
            raise ImportError(
                "AllenNLP dependencies not found. Install with: "
                "pip install bridgenlp[allennlp]"
            )
        
        self.model_name = model_name
        self.device = device
        self.aligner = TokenAligner()
        self._predictor = None
    
    @property
    @functools.lru_cache(maxsize=1)
    def predictor(self):
        """
        Lazy-load the AllenNLP predictor.
        
        Returns:
            Loaded AllenNLP predictor
        """
        from allennlp_models.coref import CorefPredictor
        
        if self._predictor is None:
            self._predictor = CorefPredictor.from_path(
                self.model_name,
                cuda_device=self.device
            )
        return self._predictor
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text with coreference resolution.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing coreference clusters
        """
        if not text.strip():
            return BridgeResult(tokens=[])
        
        # Run the model
        result = self.predictor.predict(document=text)
        
        # Extract tokens and clusters
        tokens = result.get("document", [])
        clusters = result.get("clusters", [])
        
        return BridgeResult(
            tokens=tokens,
            clusters=clusters
        )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text with coreference resolution.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing coreference clusters
        """
        text = " ".join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and attach coreference clusters.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The same Doc with coreference clusters attached
        """
        # Get raw text from the document
        text = doc.text
        
        # Process with the model
        result = self.from_text(text)
        
        # Align clusters to spaCy token boundaries
        aligned_clusters = []
        for cluster in result.clusters:
            aligned_cluster = []
            for start, end in cluster:
                # Convert to character offsets using the model's tokenization
                char_start = sum(len(result.tokens[i]) + 1 for i in range(start))
                char_end = char_start + len(result.tokens[end - 1])
                
                # Align to spaCy tokens
                span = self.aligner.align_char_span(doc, char_start, char_end)
                if span is not None:
                    aligned_cluster.append((span.start, span.end))
            
            if aligned_cluster:
                aligned_clusters.append(aligned_cluster)
        
        # Create a new result with aligned clusters
        aligned_result = BridgeResult(
            tokens=[t.text for t in doc],
            clusters=aligned_clusters
        )
        
        # Attach to the document
        return aligned_result.attach_to_spacy(doc)
    
    def __del__(self):
        """
        Clean up resources when the object is deleted.
        
        This method ensures that the model is properly unloaded
        to prevent memory leaks.
        """
        # Clear the cached predictor to free memory
        if hasattr(self, '_predictor') and self._predictor is not None:
            # Clear any GPU memory if applicable
            try:
                import torch
                if hasattr(self._predictor, 'model') and hasattr(self._predictor.model, 'cuda_device'):
                    if self._predictor.model.cuda_device >= 0:
                        torch.cuda.empty_cache()
            except (ImportError, AttributeError):
                pass
            
            # Remove references to large objects
            if hasattr(self._predictor, 'model'):
                self._predictor.model = None
            
            self._predictor = None
