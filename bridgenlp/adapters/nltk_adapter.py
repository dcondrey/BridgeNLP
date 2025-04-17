"""
NLTK adapter for BridgeNLP.

This adapter provides integration with the Natural Language Toolkit (NLTK),
allowing NLTK-processed text to be used with BridgeNLP.
"""

import functools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import spacy
from spacy.tokens import Doc

from ..aligner import TokenAligner
from ..base import BridgeBase
from ..result import BridgeResult


class NLTKBridge(BridgeBase):
    """
    Bridge adapter for NLTK.
    
    This adapter integrates NLTK's NLP capabilities with the BridgeNLP
    framework, allowing for consistent access to NLTK's functionality.
    """
    
    def __init__(self, use_pos: bool = True, use_ner: bool = True):
        """
        Initialize the NLTK bridge.
        
        Args:
            use_pos: Whether to use NLTK's part-of-speech tagging
            use_ner: Whether to use NLTK's named entity recognition
        
        Raises:
            ImportError: If NLTK is not installed
        """
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.tag import pos_tag
            from nltk.chunk import ne_chunk
        except ImportError:
            raise ImportError(
                "NLTK not found. Install with: pip install nltk"
            )
        
        # Download required NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        if use_pos:
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
        
        if use_ner:
            try:
                nltk.data.find('chunkers/maxent_ne_chunker')
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('maxent_ne_chunker')
                nltk.download('words')
        
        self.use_pos = use_pos
        self.use_ner = use_ner
        self.aligner = TokenAligner()
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text with NLTK.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing NLTK processing results
        """
        if not text.strip():
            return BridgeResult(tokens=[])
        
        import nltk
        from nltk.tokenize import word_tokenize
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Initialize results
        labels = ["O"] * len(tokens)
        spans = []
        
        # Apply POS tagging if requested
        pos_tags = []
        if self.use_pos:
            from nltk.tag import pos_tag
            pos_tags = pos_tag(tokens)
            
            # Store POS tags in the labels field
            labels = [tag for _, tag in pos_tags]
        
        # Apply NER if requested
        if self.use_ner:
            from nltk.chunk import ne_chunk
            
            # Get named entities
            if pos_tags:
                # Use existing POS tags
                ne_tree = ne_chunk(pos_tags)
            else:
                # Generate POS tags for NER
                ne_tree = ne_chunk(pos_tag(tokens))
            
            # Extract named entities
            entity_spans = []
            
            for i, chunk in enumerate(ne_tree):
                if isinstance(chunk, nltk.tree.Tree):
                    # This is a named entity
                    entity_type = chunk.label()
                    entity_tokens = [word for word, tag in chunk.leaves()]
                    
                    try:
                        # Find the start and end indices
                        leaves_idx = [j for j, node in enumerate(ne_tree) 
                                     if not isinstance(node, nltk.tree.Tree) or node == chunk]
                        
                        if not leaves_idx:
                            continue
                            
                        start_idx = leaves_idx[0]
                        end_idx = start_idx + len(entity_tokens)
                        
                        # Add the span and update labels
                        entity_spans.append((start_idx, end_idx))
                    except Exception as e:
                        warnings.warn(f"Error processing entity: {e}")
                        continue
                    for j in range(start_idx, end_idx):
                        if j < len(labels):
                            labels[j] = entity_type
            
            # Add the spans to the result
            spans.extend(entity_spans)
        
        return BridgeResult(
            tokens=tokens,
            spans=spans,
            labels=labels
        )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text with NLTK.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing NLTK processing results
        """
        import nltk
        
        # Initialize results
        labels = ["O"] * len(tokens)
        spans = []
        
        # Apply POS tagging if requested
        pos_tags = []
        if self.use_pos:
            from nltk.tag import pos_tag
            pos_tags = pos_tag(tokens)
            
            # Store POS tags in the labels field
            labels = [tag for _, tag in pos_tags]
        
        # Apply NER if requested
        if self.use_ner:
            from nltk.chunk import ne_chunk
            
            # Get named entities
            if pos_tags:
                # Use existing POS tags
                ne_tree = ne_chunk(pos_tags)
            else:
                # Generate POS tags for NER
                ne_tree = ne_chunk(pos_tag(tokens))
            
            # Extract named entities
            entity_spans = []
            
            for i, chunk in enumerate(ne_tree):
                if isinstance(chunk, nltk.tree.Tree):
                    # This is a named entity
                    entity_type = chunk.label()
                    entity_tokens = [word for word, tag in chunk.leaves()]
                    
                    # Find the start and end indices
                    leaves_idx = [j for j, node in enumerate(ne_tree) 
                                 if not isinstance(node, nltk.tree.Tree) or node == chunk]
                    start_idx = leaves_idx[0]
                    end_idx = start_idx + len(entity_tokens)
                    
                    # Add the span and update labels
                    entity_spans.append((start_idx, end_idx))
                    for j in range(start_idx, end_idx):
                        if j < len(labels):
                            labels[j] = entity_type
            
            # Add the spans to the result
            spans.extend(entity_spans)
        
        return BridgeResult(
            tokens=tokens,
            spans=spans,
            labels=labels
        )
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc with NLTK and attach the results.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The same Doc with NLTK results attached
        """
        # Extract tokens from the spaCy Doc
        tokens = [t.text for t in doc]
        
        # Process with NLTK
        result = self.from_tokens(tokens)
        
        # Attach to the document
        return result.attach_to_spacy(doc)
