"""
spaCy named entity recognition adapter for BridgeNLP.
"""

import functools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.tokens import Doc

from ..base import BridgeBase
from ..config import BridgeConfig
from ..result import BridgeResult


class SpacyNERBridge(BridgeBase):
    """
    Bridge adapter for spaCy's named entity recognition models.
    
    This adapter integrates spaCy's NER capabilities with the BridgeNLP
    framework, allowing for consistent access to entity information.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm", config: Optional[BridgeConfig] = None):
        """
        Initialize the named entity recognition bridge.
        
        Args:
            model_name: Name of the spaCy model to use
            config: Configuration for the adapter
        
        Raises:
            ImportError: If spaCy model is not installed
        """
        # Always call the parent constructor first
        super().__init__(config)
        
        # Store model name, using config if provided
        self.model_name = model_name or (config.model_name if config else None) or "en_core_web_sm"
        
        # Extract configuration options
        self.use_gpu = False
        if config and hasattr(config, "device"):
            if isinstance(config.device, str) and config.device == "cuda":
                self.use_gpu = True
            elif isinstance(config.device, int) and config.device >= 0:
                self.use_gpu = True
        
        # Initialize model
        try:
            # Set GPU preference if requested
            if self.use_gpu:
                spacy.prefer_gpu()
                
            try:
                # Load model with only NER component for efficiency
                self._nlp = spacy.load(self.model_name, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
            except OSError:
                # Attempt to download the model automatically
                try:
                    import sys
                    from spacy.cli import download
                    print(f"Downloading spaCy model '{self.model_name}'... (this may take a moment)")
                    download(self.model_name)
                    self._nlp = spacy.load(self.model_name, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
                    print(f"Successfully downloaded and loaded spaCy model '{self.model_name}'")
                except Exception as e:
                    raise ImportError(
                        f"spaCy model '{self.model_name}' not found and automatic download failed: {str(e)}. "
                        f"Install manually with: python -m spacy download {self.model_name}"
                    )
            
            # Verify NER component is available
            if "ner" not in self._nlp.pipe_names:
                raise ValueError(f"Model {self.model_name} does not have an NER component")
                
        except Exception as e:
            # Catch any other exceptions that might occur during initialization
            if isinstance(e, ImportError) or isinstance(e, ValueError):
                # Re-raise the specific errors we've already captured
                raise e
            else:
                # For other exceptions, provide a more generic error message
                raise RuntimeError(
                    f"Error initializing spaCy NER model '{self.model_name}': {str(e)}"
                )
    
    @property
    def nlp(self):
        """
        Access the spaCy model.
        
        Returns:
            Loaded spaCy model
        """
        return self._nlp
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text with named entity recognition.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing named entities
        """
        with self._measure_performance():
            if not text.strip():
                return BridgeResult(tokens=[])
            
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract tokens and entities
            tokens = [token.text for token in doc]
            labels = ["O"] * len(tokens)  # Default to outside any entity
            spans = []
            
            # Convert entities to spans and labels
            for ent in doc.ents:
                spans.append((ent.start, ent.end))
                # Set entity labels for tokens
                for i in range(ent.start, ent.end):
                    labels[i] = ent.label_
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(tokens)
            
            return BridgeResult(
                tokens=tokens,
                spans=spans,
                labels=labels
            )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text with named entity recognition.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing named entities
        """
        with self._measure_performance():
            # Create a Doc object from the tokens
            doc = Doc(self.nlp.vocab, words=tokens)
            
            # Process the doc with the NER pipeline
            for name, proc in self.nlp.pipeline:
                if name == "ner":
                    doc = proc(doc)
            
            # Extract entity spans and labels
            spans = []
            labels = ["O"] * len(tokens)  # Default to outside any entity
            
            # Convert entities to spans and labels
            for ent in doc.ents:
                spans.append((ent.start, ent.end))
                # Set entity labels for tokens
                for i in range(ent.start, ent.end):
                    labels[i] = ent.label_
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(tokens)
            
            return BridgeResult(
                tokens=tokens,
                spans=spans,
                labels=labels
            )
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and attach named entity information.
        
        If the Doc already has entities, they will be preserved.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The same Doc with named entity information attached
        """
        with self._measure_performance():
            # If the doc already has entities, use those
            if len(doc.ents) > 0:
                spans = [(ent.start, ent.end) for ent in doc.ents]
                labels = ["O"] * len(doc)
                
                for ent in doc.ents:
                    for i in range(ent.start, ent.end):
                        labels[i] = ent.label_
                
                result = BridgeResult(
                    tokens=[t.text for t in doc],
                    spans=spans,
                    labels=labels
                )
                
                # Update token count for metrics
                self._metrics["total_tokens"] += len(doc)
                
                return result.attach_to_spacy(doc)
            
            # Otherwise, process with our NER model
            # We need to create a new Doc to avoid modifying the original
            processed_doc = self.nlp(doc.text)
            
            # Extract entities
            spans = [(ent.start, ent.end) for ent in processed_doc.ents]
            labels = ["O"] * len(doc)
            
            # Map entities back to original doc
            for ent in processed_doc.ents:
                # Find the corresponding span in the original doc
                start_char = processed_doc[ent.start].idx
                end_char = processed_doc[ent.end - 1].idx + len(processed_doc[ent.end - 1])
                
                # Find tokens in original doc that correspond to this span
                start_token = None
                end_token = None
                
                for i, token in enumerate(doc):
                    if token.idx <= start_char < token.idx + len(token.text) and start_token is None:
                        start_token = i
                    if token.idx <= end_char <= token.idx + len(token.text):
                        end_token = i + 1
                        break
                
                if start_token is not None and end_token is not None:
                    # Check if this span already exists
                    if (start_token, end_token) not in spans:
                        spans.append((start_token, end_token))
                        for i in range(start_token, end_token):
                            labels[i] = ent.label_
            
            result = BridgeResult(
                tokens=[t.text for t in doc],
                spans=spans,
                labels=labels
            )
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(doc)
            
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
            self._nlp = None
