"""
spaCy named entity recognition adapter for BridgeNLP.
"""

import functools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.tokens import Doc

from ..base import BridgeBase
from ..result import BridgeResult


class SpacyNERBridge(BridgeBase):
    """
    Bridge adapter for spaCy's named entity recognition models.
    
    This adapter integrates spaCy's NER capabilities with the BridgeNLP
    framework, allowing for consistent access to entity information.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the named entity recognition bridge.
        
        Args:
            model_name: Name of the spaCy model to use
        
        Raises:
            ImportError: If spaCy model is not installed
        """
        try:
            self.nlp = spacy.load(model_name, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
            # Only enable NER component for efficiency
            if "ner" not in self.nlp.pipe_names:
                raise ValueError(f"Model {model_name} does not have an NER component")
        except OSError:
            raise ImportError(
                f"spaCy model '{model_name}' not found. Install with: "
                f"python -m spacy download {model_name}"
            )
        
        self.model_name = model_name
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text with named entity recognition.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing named entities
        """
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
        text = " ".join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and attach named entity information.
        
        If the Doc already has entities, they will be preserved.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The same Doc with named entity information attached
        """
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
        
        return result.attach_to_spacy(doc)
