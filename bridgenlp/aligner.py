"""
Token alignment utilities for mapping between different tokenization schemes.
"""

import re
import warnings
from typing import List, Optional, Tuple

import spacy
from spacy.tokens import Doc, Span


class TokenAligner:
    """
    Utility for aligning tokens between different tokenization schemes.
    
    This class provides methods to map character spans and token indices
    between different tokenization schemes, such as between model-specific
    tokenizers and spaCy's tokenizer.
    """
    
    def align_char_span(self, doc: Doc, start_char: int, end_char: int) -> Optional[Span]:
        """
        Align a character span to spaCy token boundaries.
        
        Args:
            doc: spaCy Doc to align with
            start_char: Character offset for span start
            end_char: Character offset for span end
            
        Returns:
            spaCy Span object or None if alignment fails
        """
        if doc is None:
            warnings.warn("Cannot align span: Doc is None")
            return None
            
        if start_char < 0 or end_char > len(doc.text) or start_char >= end_char:
            warnings.warn(f"Invalid character span: ({start_char}, {end_char})")
            return None
        
        # Find tokens that contain the start and end characters
        start_token = None
        end_token = None
        
        for i, token in enumerate(doc):
            # Find start token
            if token.idx <= start_char < token.idx + len(token.text) and start_token is None:
                start_token = i
            
            # Find end token (the token that contains the last character)
            if token.idx <= end_char <= token.idx + len(token.text):
                # If end_char is at token boundary, we want the previous token
                if end_char == token.idx and i > 0:
                    end_token = i - 1
                else:
                    end_token = i
                break
        
        if start_token is not None and end_token is not None:
            # Add 1 to end_token to make it exclusive for Span creation
            return doc[start_token:end_token + 1]
        
        warnings.warn(f"Failed to align character span ({start_char}, {end_char})")
        return None
    
    def align_token_span(self, doc: Doc, start_idx: int, end_idx: int, 
                         model_tokens: List[str]) -> Optional[Span]:
        """
        Align a token span from a different tokenization to spaCy tokens.
        
        Args:
            doc: spaCy Doc to align with
            start_idx: Start index in model's tokenization
            end_idx: End index in model's tokenization
            model_tokens: The tokens from the model's tokenization
            
        Returns:
            spaCy Span object or None if alignment fails
        """
        if start_idx < 0 or end_idx > len(model_tokens) or start_idx >= end_idx:
            warnings.warn(f"Invalid token span: ({start_idx}, {end_idx})")
            return None
        
        # Reconstruct the text from model tokens
        span_text = " ".join(model_tokens[start_idx:end_idx])
        
        # Use fuzzy alignment as a fallback
        return self.fuzzy_align(doc, span_text)
    
    def fuzzy_align(self, doc: Doc, text_segment: str) -> Optional[Span]:
        """
        Find the best matching span in a document for a given text segment.
        
        This method uses an optimized approach for large documents by:
        1. First trying exact string matching
        2. Using a sliding window approach for fuzzy matching
        3. Limiting the search space based on segment length
        
        Args:
            doc: spaCy Doc to search in
            text_segment: Text to find in the document
            
        Returns:
            spaCy Span object or None if no good match is found
        """
        # Clean up the text segment for better matching
        clean_segment = re.sub(r'\s+', ' ', text_segment).strip()
        if not clean_segment:
            warnings.warn("Empty text segment for alignment")
            return None
        
        # Simple case: exact match
        doc_text = doc.text.lower()
        segment_lower = clean_segment.lower()
        
        # Try exact match first
        start_char = doc_text.find(segment_lower)
        if start_char >= 0:
            end_char = start_char + len(segment_lower)
            return self.align_char_span(doc, start_char, end_char)
        
        # For large documents, use a more efficient approach
        if len(doc) > 1000:
            return self._fuzzy_align_large_doc(doc, clean_segment)
        
        # For smaller documents, use the more thorough approach
        segment_tokens = clean_segment.split()
        best_match = None
        best_score = 0
        
        # Limit the search space based on segment length
        max_window_size = min(len(segment_tokens) * 3, len(doc))
        
        for i in range(len(doc)):
            for j in range(i + 1, min(i + max_window_size, len(doc) + 1)):
                span = doc[i:j]
                span_text = span.text.lower()
                
                # Calculate similarity score (token overlap with position weighting)
                span_tokens = span_text.split()
                
                # Skip if length difference is too large
                if abs(len(span_tokens) - len(segment_tokens)) > max(2, len(segment_tokens) // 2):
                    continue
                
                # Calculate token overlap with position weighting
                common = set(span_tokens).intersection(segment_tokens)
                
                # Base score on token overlap
                score = len(common) / max(len(span_tokens), len(segment_tokens))
                
                # Bonus for position-preserving matches
                position_bonus = 0
                for pos, token in enumerate(segment_tokens):
                    if pos < len(span_tokens) and span_tokens[pos].lower() == token.lower():
                        position_bonus += 0.1
                
                score += position_bonus
                
                if score > best_score and score > 0.5:  # Threshold for acceptable match
                    best_score = score
                    best_match = span
        
        if best_match is None:
            warnings.warn(f"Failed to align text segment: '{text_segment}'")
        
        return best_match
    
    def _fuzzy_align_large_doc(self, doc: Doc, text_segment: str) -> Optional[Span]:
        """
        Optimized fuzzy alignment for large documents.
        
        Uses a multi-stage approach for efficient alignment:
        1. First tries a hash-based search for exact matches
        2. Then uses a sliding window with early stopping
        3. Finally uses a more targeted search in promising regions
        
        Args:
            doc: spaCy Doc to search in
            text_segment: Text to find in the document
            
        Returns:
            spaCy Span object or None if no good match is found
        """
        segment_tokens = text_segment.lower().split()
        segment_len = len(segment_tokens)
        
        if segment_len == 0:
            return None
        
        # Create token sets for faster lookup
        segment_token_set = set(segment_tokens)
        
        # Stage 1: Try to find distinctive tokens that might be unique
        # Focus on longer tokens which are more likely to be distinctive
        distinctive_tokens = sorted(
            [t for t in segment_tokens if len(t) > 4],
            key=len, reverse=True
        )[:3]  # Take up to 3 longest tokens
        
        if distinctive_tokens:
            # Create a quick index of these tokens in the document
            token_positions = {}
            for i, token in enumerate(doc):
                token_lower = token.text.lower()
                if token_lower in distinctive_tokens:
                    if token_lower not in token_positions:
                        token_positions[token_lower] = []
                    token_positions[token_lower].append(i)
            
            # Check if we found any of our distinctive tokens
            candidate_positions = []
            for token in distinctive_tokens:
                if token in token_positions:
                    candidate_positions.extend(token_positions[token])
            
            # If we found positions, check around them first
            if candidate_positions:
                best_match = None
                best_score = 0
                
                for pos in candidate_positions:
                    # Check a window around this position
                    start_idx = max(0, pos - segment_len)
                    end_idx = min(len(doc), pos + segment_len)
                    
                    # Try different spans within this focused window
                    for j in range(start_idx, pos + 1):
                        for k in range(pos + 1, end_idx + 1):
                            if k - j < segment_len / 2 or k - j > segment_len * 2:
                                continue  # Skip if span length is too different
                            
                            span = doc[j:k]
                            span_text = span.text.lower()
                            span_tokens = span_text.split()
                            
                            # Calculate similarity score with optimized method
                            common = segment_token_set.intersection(span_tokens)
                            score = len(common) / max(len(span_tokens), segment_len)
                            
                            # Early stopping: if we find a very good match, return immediately
                            if score > 0.8:
                                return span
                            
                            if score > best_score and score > 0.5:
                                best_score = score
                                best_match = span
                
                # If we found a good match in our focused search, return it
                if best_match is not None and best_score > 0.6:
                    return best_match
        
        # Stage 2: If we didn't find a good match with distinctive tokens,
        # fall back to an optimized sliding window approach
        best_match = None
        best_score = 0
        
        # Use adaptive window and step sizes based on document and segment length
        window_size = min(segment_len * 3, 20)
        step_size = max(1, window_size // 3)
        
        # For very large documents, increase step size further
        if len(doc) > 10000:
            step_size = max(5, step_size)
        
        # Slide through the document with the window
        for i in range(0, len(doc), step_size):
            # Create a window of tokens
            end_idx = min(i + window_size, len(doc))
            window_tokens = [t.text.lower() for t in doc[i:end_idx]]
            
            # Quick check if there's any overlap in tokens
            window_token_set = set(window_tokens)
            overlap = segment_token_set.intersection(window_token_set)
            
            # Skip windows with insufficient overlap
            overlap_threshold = max(1, segment_len // 3)
            if len(overlap) < overlap_threshold:
                continue
            
            # Check different spans within this window, but be smarter about which ones we check
            # Start with spans that are closest in length to our target
            span_candidates = []
            for j in range(i, end_idx):
                for k in range(j + 1, min(j + segment_len * 2, end_idx + 1)):
                    span_len = k - j
                    # Skip if length difference is too large
                    if abs(span_len - segment_len) > max(2, segment_len // 2):
                        continue
                    
                    # Calculate a priority score based on how close the length is to our target
                    length_diff = abs(span_len - segment_len)
                    priority = 1.0 / (1.0 + length_diff)
                    span_candidates.append((j, k, priority))
            
            # Sort candidates by priority (highest first)
            span_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Check the most promising candidates first, with early stopping
            for j, k, _ in span_candidates[:10]:  # Limit to top 10 candidates
                span = doc[j:k]
                span_text = span.text.lower()
                span_tokens = span_text.split()
                
                # Calculate similarity score
                common = segment_token_set.intersection(span_tokens)
                score = len(common) / max(len(span_tokens), segment_len)
                
                # Bonus for position-preserving matches
                position_bonus = 0
                for pos, token in enumerate(segment_tokens):
                    if pos < len(span_tokens) and span_tokens[pos] == token:
                        position_bonus += 0.1
                
                score += position_bonus
                
                # Early stopping for excellent matches
                if score > 0.8:
                    return span
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = span
        
        return best_match
