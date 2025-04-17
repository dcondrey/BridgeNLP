"""
Hugging Face question answering adapter for BridgeNLP.
"""

import functools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.tokens import Doc

from ..aligner import TokenAligner
from ..base import BridgeBase
from ..result import BridgeResult


class HuggingFaceQABridge(BridgeBase):
    """
    Bridge adapter for Hugging Face's question answering models.
    
    This adapter integrates transformer-based QA models from Hugging Face
    with token-based pipelines like spaCy.
    """
    
    def __init__(self, model_name: str = "deepset/roberta-base-squad2", 
                 device: int = -1):
        """
        Initialize the question answering bridge.
        
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
        self._current_question = None
    
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
                "question-answering", 
                model=self.model_name,
                device=device
            )
        return self._pipeline
    
    def set_question(self, question: str) -> None:
        """
        Set the question to be used for QA.
        
        Args:
            question: The question to answer
            
        Raises:
            ValueError: If question is empty or None
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        self._current_question = question.strip()
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text with question answering.
        
        Args:
            text: Raw text to process (context)
            
        Returns:
            BridgeResult containing question answering results
        """
        if not text.strip():
            return BridgeResult(tokens=[])
        
        if not self._current_question:
            warnings.warn("No question set for QA. Using default question.")
            self._current_question = "What is this text about?"
        
        # Run the model
        result = self.pipeline(
            question=self._current_question,
            context=text
        )
        
        # Extract tokens (we'll use a simple whitespace tokenizer for now)
        tokens = text.split()
        
        # Process the QA result
        roles = [{
            "role": "ANSWER",
            "text": result["answer"],
            "score": result["score"],
            "start": result["start"],
            "end": result["end"]
        }]
        
        return BridgeResult(
            tokens=tokens,
            roles=roles
        )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text with question answering.
        
        Args:
            tokens: List of pre-tokenized strings (context)
            
        Returns:
            BridgeResult containing question answering results
        """
        text = " ".join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and attach question answering results.
        
        Args:
            doc: spaCy Doc to process (context)
            
        Returns:
            The same Doc with question answering results attached
        """
        if not self._current_question:
            warnings.warn("No question set for QA. Use set_question() first.")
            return doc
        
        # Get raw text from the document
        text = doc.text
        
        # Process with the model
        result = self.from_text(text)
        
        # Align answer to spaCy token boundaries
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
