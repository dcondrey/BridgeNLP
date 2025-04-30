"""
Bridge adapter for Hugging Face's embedding models.

This adapter integrates transformer-based embedding models from Hugging Face
with token-based pipelines like spaCy.
"""

import functools
from typing import Dict, List, Optional, Union

import numpy as np
import spacy
from spacy.tokens import Doc

from ..base import BridgeBase
from ..config import BridgeConfig
from ..result import BridgeResult


class HuggingFaceEmbeddingsBridge(BridgeBase):
    """
    Bridge adapter for Hugging Face's embedding models.
    
    This adapter integrates transformer-based embedding models from Hugging Face
    with token-based pipelines like spaCy, providing vector representations
    for text and tokens.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: Union[int, str] = -1, config: Optional[BridgeConfig] = None):
        """
        Initialize the bridge adapter.
        
        Args:
            model_name: Name or path of the Hugging Face model to use
            device: Device to run the model on (-1 for CPU, >=0 for specific GPU, or "cuda"/"cpu")
            config: Configuration for the adapter
        """
        # Always call the parent constructor first
        super().__init__(config)
        
        # Store model name, using config if provided
        self.model_name = model_name or (config.model_name if config else None) or "sentence-transformers/all-MiniLM-L6-v2"
        
        # Extract configuration options with defaults
        self.device = config.device if config else device
        self.max_length = config.max_length if config and config.max_length else 512
        self.cache_size = config.cache_size if config and config.cache_size else 1
        
        # Additional parameters from config
        if config and config.params:
            self.pooling = config.params.get("pooling", "mean")
            self.normalize = config.params.get("normalize", True)
        else:
            self.pooling = "mean"
            self.normalize = True
        
        # Initialize model lazily
        self._model = None
        self._tokenizer = None
    
    @property
    def model(self):
        """
        Lazily load the embedding model.
        
        Returns:
            Hugging Face model for embeddings
        """
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                # Load tokenizer and model
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                
                # Move to specified device
                if isinstance(self.device, str):
                    if self.device == "cuda" and torch.cuda.is_available():
                        self._model = self._model.cuda()
                    elif self.device != "-1" and self.device != "cpu" and torch.cuda.is_available():
                        # Try to parse as integer for specific GPU
                        try:
                            device_idx = int(self.device)
                            if device_idx >= 0:
                                self._model = self._model.cuda(device_idx)
                        except ValueError:
                            # If not a valid integer, default to CPU
                            pass
                elif isinstance(self.device, int) and self.device >= 0 and torch.cuda.is_available():
                    self._model = self._model.cuda(self.device)
                
                # Set to evaluation mode
                self._model.eval()
                
            except ImportError:
                raise ImportError(
                    "Hugging Face transformers not installed. "
                    "Install with: pip install transformers torch"
                )
        
        return self._model
    
    def _get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embeddings for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        import torch
        
        # Ensure model and tokenizer are loaded
        model = self.model  # This will load the model if not already loaded
        
        # Tokenize
        inputs = self._tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Move to device
        if isinstance(self.device, str):
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device != "-1" and self.device != "cpu" and torch.cuda.is_available():
                # Try to parse as integer for specific GPU
                try:
                    device_idx = int(self.device)
                    if device_idx >= 0:
                        inputs = {k: v.cuda(device_idx) for k, v in inputs.items()}
                except ValueError:
                    # If not a valid integer, default to CPU
                    pass
        elif isinstance(self.device, int) and self.device >= 0 and torch.cuda.is_available():
            inputs = {k: v.cuda(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get embeddings from the model output
        embeddings = outputs.last_hidden_state
        
        # Apply pooling
        if self.pooling == "mean":
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask
            summed = torch.sum(masked_embeddings, 1)
            counts = torch.clamp(torch.sum(mask, 1), min=1e-9)
            pooled = summed / counts
        elif self.pooling == "cls":
            # CLS token pooling
            pooled = embeddings[:, 0]
        else:
            # Default to mean pooling
            attention_mask = inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask
            summed = torch.sum(masked_embeddings, 1)
            counts = torch.clamp(torch.sum(mask, 1), min=1e-9)
            pooled = summed / counts
        
        # Normalize if requested
        if self.normalize:
            pooled = pooled / pooled.norm(dim=1, keepdim=True)
        
        # Convert to numpy
        return pooled.cpu().numpy()
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text and return embedding results.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing embeddings
        """
        with self._measure_performance():
            # Get embeddings
            embeddings = self._get_embeddings(text)
            
            # Create tokens (simple whitespace tokenization for now)
            tokens = text.split()
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(tokens)
            
            # Store embeddings in the roles field (as it's a flexible dict)
            roles = [{"embedding": embeddings[0].tolist()}]
            
            # Return standardized result
            return BridgeResult(
                tokens=tokens,
                roles=roles
            )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text and return embedding results.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing embeddings
        """
        with self._measure_performance():
            # Reconstruct text from tokens
            text = " ".join(tokens)
            
            # Get embeddings
            embeddings = self._get_embeddings(text)
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(tokens)
            
            # Store embeddings in the roles field (as it's a flexible dict)
            roles = [{"embedding": embeddings[0].tolist()}]
            
            # Return standardized result
            return BridgeResult(
                tokens=tokens,
                roles=roles
            )
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and return an enhanced Doc with embeddings.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            The same Doc with embedding attributes attached
        """
        with self._measure_performance():
            # Extract tokens from the doc
            tokens = [token.text for token in doc]
            
            # Process using the from_tokens method
            result = self.from_tokens(tokens)
            
            # Attach results to the doc
            return result.attach_to_spacy(doc)
    
    def cleanup(self):
        """
        Clean up resources used by this adapter.
        
        This method is called when the adapter is used as a context manager
        or when it's garbage collected.
        """
        # Unload the model if requested in config
        if (hasattr(self, "config") and self.config and 
            hasattr(self.config, "unload_on_del") and self.config.unload_on_del):
            import gc
            self._model = None
            self._tokenizer = None
            gc.collect()
