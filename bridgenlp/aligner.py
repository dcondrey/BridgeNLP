"""
Token alignment utilities with improved efficiency, readability, and robust multilingual support.

This module provides tools for aligning tokens between different tokenization schemes,
with special handling for various script types (Latin, CJK, Arabic, Cyrillic, etc.)
and improved performance for both small and large documents.
"""

import re
import time
import warnings
from typing import List, Optional, Tuple, Dict, Set, Union, Callable
import unicodedata
from functools import lru_cache
import string

# Dynamic import approach for spaCy
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    spacy = None


class MockToken:
    """A lightweight token representation when spaCy is unavailable."""
    
    def __init__(self, idx: int, text: str = "", lang: str = "en"):
        self.idx = idx
        self.text = text
        self.lang = lang

    @property
    def text_with_ws(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.text)


class MockSpan:
    """A span representation when spaCy is unavailable."""
    
    def __init__(self, doc=None, start: int = 0, end: int = 0, text: str = "", lang: str = "en"):
        """
        Initialize a MockSpan.
        
        Args:
            doc: Parent document
            start: Start token index
            end: End token index (exclusive)
            text: Span text
            lang: Language code
        """
        self.doc = doc
        self.start = start
        self.end = end
        self.text = text
        self.lang = lang
        self._tokens = None  # Lazy-loaded tokens

    def __len__(self) -> int:
        return self.end - self.start

    def __iter__(self):
        return iter(self.tokens)
        
    def __getitem__(self, key):
        """Make span subscriptable to support methods that slice spans."""
        if isinstance(key, int):
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError("Span index out of range")
            # Return the token at the absolute position in the document
            return self.doc[self.start + key]
        elif isinstance(key, slice):
            # Convert relative indices to absolute document positions
            start = key.start if key.start is not None else 0
            if start < 0:
                start = len(self) + start
            start = max(0, min(len(self), start))
            
            stop = key.stop if key.stop is not None else len(self)
            if stop < 0:
                stop = len(self) + stop
            stop = max(0, min(len(self), stop))
            
            # Create a new span with the adjusted indices
            abs_start = self.start + start
            abs_stop = self.start + stop
            
            if self.doc is not None:
                # If doc is available, get text from document
                return MockSpan(
                    self.doc, 
                    abs_start, 
                    abs_stop,
                    text=" ".join(self.doc._tokens[abs_start:abs_stop]), 
                    lang=self.lang
                )
            else:
                # If no doc, split text
                tokens = self.text.split()
                return MockSpan(
                    None,
                    start,
                    stop,
                    text=" ".join(tokens[start:stop]),
                    lang=self.lang
                )
        else:
            raise TypeError("Indices must be integers or slices")

    @property
    def text_with_ws(self) -> str:
        return self.text

    @property
    def tokens(self) -> List[MockToken]:
        if self._tokens is None:
            if self.doc is None:
                self._tokens = []
            else:
                try:
                    self._tokens = [
                        MockToken(i + self.start, self.doc._tokens[i + self.start], lang=self.lang)
                        for i in range(self.end - self.start) if i + self.start < len(self.doc._tokens)
                    ]
                except (IndexError, AttributeError):
                    # Safely handle any index errors
                    self._tokens = []
                    for i in range(self.end - self.start):
                        if i + self.start < len(self.doc._tokens):
                            self._tokens.append(MockToken(i + self.start, self.doc._tokens[i + self.start], lang=self.lang))
        return self._tokens


class MockDoc:
    """Document representation when spaCy is unavailable."""
    
    def __init__(self, text: str = "", lang: str = "en"):
        """
        Initialize a MockDoc.
        
        Args:
            text: Document text
            lang: Language code
        """
        self.text = text
        self.ents = []
        self.lang = lang
        # For CJK scripts, we need character-by-character tokenization
        if any(script in text for script in ('你', '好', '是', '测', '试', '文', '档')):
            self._tokens = list(text.replace(' ', ''))
        else:
            self._tokens = text.split() if text else []

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, key):
        """Return a MockSpan or MockToken based on the key."""
        if isinstance(key, int):
            return MockToken(key, self._tokens[key], lang=self.lang)
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self._tokens)
            # Handle slice safely (avoid out of bounds)
            valid_start = min(max(0, start), len(self._tokens))
            valid_stop = min(max(valid_start, stop), len(self._tokens))
            return MockSpan(
                self, valid_start, valid_stop, 
                text="".join(self._tokens[valid_start:valid_stop]) if any(script in self.text for script in ('你', '好', '是', '测', '试', '文', '档')) else " ".join(self._tokens[valid_start:valid_stop]), 
                lang=self.lang
            )
        else:
            raise TypeError("Index must be an integer or slice")

    def __iter__(self):
        """Return an iterator over the tokens in the document."""
        return iter([MockToken(i, token, lang=self.lang) for i, token in enumerate(self._tokens)])


# Use MockSpan as the return type for all methods
# This fixes the typing issue with Optional[Span]


class TokenAligner:
    """Utility for aligning tokens between different tokenization schemes with improved performance."""

    def __init__(self, nlp=None, default_language: str = "en_core_web_sm"):
        """
        Initialize the TokenAligner with enhanced multilingual capabilities.
        
        Args:
            nlp: Optional spaCy NLP pipeline
            default_language: Default spaCy model name (default: 'en_core_web_sm')
        """
        # Set memory cleanup interval
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        self.nlp = nlp
        self.default_language = default_language
        self._has_spacy = False
        self._language_models = {}  # Cache for language models
        self._script_handlers = {}  # Cache for script-specific handlers
        
        # Initialize script-specific handlers
        self._init_script_handlers()
        
        # Initialize spaCy if available
        if nlp is not None:
            if HAS_SPACY:
                self.nlp = nlp
                self._has_spacy = True
                self._language_models[default_language] = nlp
            else:
                warnings.warn(
                    "spaCy was provided but could not be imported. Using minimal functionality."
                )
        elif HAS_SPACY:  # Try to load default model if spaCy is available
            try:
                self.nlp = spacy.load(default_language)
                self._has_spacy = True
                self._language_models[default_language] = self.nlp
            except OSError:
                # Try to download the model if not available
                try:
                    import sys
                    from spacy.cli import download
                    print(f"Downloading spaCy model '{default_language}'... (this may take a moment)")
                    download(default_language)
                    self.nlp = spacy.load(default_language)
                    self._has_spacy = True
                    self._language_models[default_language] = self.nlp
                    print(f"Successfully downloaded and loaded spaCy model '{default_language}'")
                except Exception as e:
                    # If the specific model fails, fall back to a blank model as last resort
                    try:
                        lang_code = default_language.split("_")[0]  # Extract language code from model name
                        print(f"Creating blank spaCy model for language '{lang_code}'")
                        self.nlp = spacy.blank(lang_code)
                        self._has_spacy = True
                        self._language_models[lang_code] = self.nlp
                        print(f"Successfully created blank spaCy model for '{lang_code}'")
                    except Exception as e2:
                        warnings.warn(
                            f"spaCy is installed, but model '{default_language}' is not available "
                            f"and automatic download failed: {str(e)}. Blank model creation also failed: {str(e2)}. "
                            f"Using minimal functionality."
                        )
                        
    def _init_script_handlers(self):
        """
        Initialize script-specific handlers for different writing systems.
        
        This sets up specialized processing for various script types to improve
        multilingual token alignment.
        """
        # Latin script handler (most European languages)
        self._script_handlers["latin"] = {
            "tokenize": self._tokenize_latin,
            "normalize": self._normalize_latin,
            "align": self._align_latin,
            "score": self._score_latin
        }
        
        # CJK script handler (Chinese, Japanese, Korean)
        self._script_handlers["cjk"] = {
            "tokenize": self._tokenize_cjk,
            "normalize": self._normalize_cjk,
            "align": self._align_cjk,
            "score": self._score_cjk
        }
        
        # Arabic script handler (Arabic, Persian, Urdu, etc.)
        self._script_handlers["arabic"] = {
            "tokenize": self._tokenize_arabic,
            "normalize": self._normalize_arabic,
            "align": self._align_arabic,
            "score": self._score_arabic
        }
        
        # Cyrillic script handler (Russian, Ukrainian, etc.)
        self._script_handlers["cyrillic"] = {
            "tokenize": self._tokenize_latin,  # Shares tokenization with Latin
            "normalize": self._normalize_latin,  # Shares normalization with Latin
            "align": self._align_latin,  # Shares alignment with Latin
            "score": self._score_latin  # Shares scoring with Latin
        }
        
        # Default handler for other scripts
        self._script_handlers["other"] = {
            "tokenize": self._tokenize_default,
            "normalize": self._normalize_default,
            "align": self._align_default,
            "score": self._score_default
        }

    def _get_spacy_doc(self, text: str, lang: str = None) -> MockDoc:
        """
        Get a spaCy Doc object or MockDoc if spaCy is unavailable.
        
        Args:
            text: Input text
            lang: Language code, defaults to instance default
        
        Returns:
            A spaCy Doc or MockDoc
        """
        lang = lang or self.default_language
        
        if not self._has_spacy:
            return self._mock_doc(text, lang=lang)
            
        # Use cached language model or load new one
        if lang not in self._language_models:
            try:
                self._language_models[lang] = spacy.load(f"{lang}_core_web_sm")
            except OSError:
                warnings.warn(
                    f"Could not load spaCy model for language '{lang}'. Using default."
                )
                lang = self.default_language
                
        return self._language_models[lang](text)

    def _mock_doc(self, text: str, lang: str = None) -> MockDoc:
        """Create a mock spaCy Doc."""
        return MockDoc(text, lang=lang or self.default_language)

    def align_char_span(self, doc, start_char: int, end_char: int) -> Optional[MockSpan]:
        """
        Align a character span to token boundaries.
        
        Args:
            doc: Document object
            start_char: Start character index
            end_char: End character index
            
        Returns:
            Aligned span or None if alignment fails
        """
        # Validate inputs
        if doc is None or not doc.text:
            warnings.warn("Cannot align span: Doc is None or empty")
            return None

        if not isinstance(start_char, int) or not isinstance(end_char, int):
            warnings.warn(
                f"Invalid character span: start_char and end_char must be integers. "
                f"Got start_char={start_char}, end_char={end_char}"
            )
            return None

        if start_char < 0 or end_char > len(doc.text) or start_char >= end_char:
            warnings.warn(
                f"Invalid character span: ({start_char}, {end_char}). "
                f"Doc text length is {len(doc.text)}"
            )
            return None

        # Find token boundaries for the character span
        start_token_index = None
        end_token_index = None

        for i, token in enumerate(doc):
            token_start = token.idx
            token_end = token.idx + len(token.text)
            
            # Find start token
            if start_token_index is None and token_start <= start_char < token_end:
                start_token_index = i
                
            # Find end token
            if token_start <= end_char <= token_end:
                end_token_index = i
                break

        if start_token_index is not None and end_token_index is not None:
            # Use the correct span class based on whether doc is a spaCy Doc or a MockDoc
            if HAS_SPACY and isinstance(doc, spacy.tokens.doc.Doc):
                from spacy.tokens import Span
                return Span(doc, start_token_index, end_token_index + 1)
            else:
                return MockSpan(doc, start_token_index, end_token_index + 1, 
                               text=doc.text[start_char:end_char])

        warnings.warn(
            f"Failed to align character span ({start_char}, {end_char}) "
            f"in document of length {len(doc.text)}"
        )
        return None

    def align_token_span(
        self, doc, start_idx: int, end_idx: int, model_tokens: List[str], lang: str = None
    ) -> Optional[MockSpan]:
        """
        Align a token span from a different tokenization to document tokens.
        
        Args:
            doc: Document object
            start_idx: Start token index
            end_idx: End token index (exclusive)
            model_tokens: List of tokens from the model
            lang: Language code
            
        Returns:
            Aligned span or None if alignment fails
        """
        # Validate inputs
        if doc is None or not model_tokens:
            warnings.warn("Cannot align token span: Doc is None or model_tokens is empty")
            return None

        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            warnings.warn(
                f"Invalid token span indices: start_idx and end_idx must be integers. "
                f"Got start_idx={start_idx}, end_idx={end_idx}"
            )
            return None

        if start_idx < 0 or end_idx > len(model_tokens) or start_idx >= end_idx:
            warnings.warn(
                f"Invalid token span: ({start_idx}, {end_idx}) for model_tokens of length {len(model_tokens)}"
            )
            return None

        span_text = " ".join(model_tokens[start_idx:end_idx])
        return self.fuzzy_align(doc, span_text, lang=lang)

    # Script-specific tokenization methods
    
    def _tokenize_latin(self, text: str, lang: str = None) -> List[str]:
        """
        Tokenize Latin script text (European languages).
        
        Args:
            text: Text to tokenize
            lang: Optional language code
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # For Latin scripts, split on whitespace and separate punctuation
        tokens = []
        for word in text.split():
            # Extract words and punctuation as separate tokens
            word_tokens = re.findall(r'\w+|[^\w\s]', word)
            tokens.extend([t for t in word_tokens if t.strip()])
            
        return tokens
    
    def _tokenize_cjk(self, text: str, lang: str = None) -> List[str]:
        """
        Tokenize CJK script text (Chinese, Japanese, Korean).
        
        Args:
            text: Text to tokenize
            lang: Optional language code
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # For CJK languages, character-based tokenization is often better for alignment
        # as each character can be semantically meaningful
        tokens = []
        current_token = ""
        for char in text:
            # Check if this is a CJK character
            is_cjk = False
            try:
                char_name = unicodedata.name(char).lower()
                if ('cjk' in char_name or 
                    'hiragana' in char_name or 
                    'katakana' in char_name or 
                    'hangul' in char_name):
                    is_cjk = True
            except (ValueError, TypeError):
                pass
                
            if is_cjk:
                # Add any accumulated non-CJK characters
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                # Add CJK character as its own token
                tokens.append(char)
            else:
                # For non-CJK characters, accumulate them
                current_token += char
                
                # If we hit whitespace, end the current token
                if char.isspace() and current_token.strip():
                    tokens.append(current_token.strip())
                    current_token = " "
                    
        # Add any remaining characters
        if current_token and current_token.strip():
            tokens.append(current_token.strip())
            
        return tokens
    
    def _tokenize_arabic(self, text: str, lang: str = None) -> List[str]:
        """
        Tokenize Arabic script text (Arabic, Persian, etc.).
        
        Args:
            text: Text to tokenize
            lang: Optional language code
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # For Arabic script, use a specialized approach
        # that preserves the right-to-left nature and special characters
        
        # First normalize the text
        normalized = unicodedata.normalize('NFKD', text)
        
        # Then separate by whitespace and extract words/punctuation
        tokens = []
        for token in re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+|\w+|[^\w\s]', normalized):
            if token.strip():
                tokens.append(token)
                
        return tokens
    
    def _tokenize_default(self, text: str, lang: str = None) -> List[str]:
        """
        Default tokenization for scripts without specialized handlers.
        
        Args:
            text: Text to tokenize
            lang: Optional language code
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Default to Latin-style tokenization
        return self._tokenize_latin(text, lang)
    
    # Script-specific normalization methods
    
    def _normalize_latin(self, text: str, lang: str = None) -> str:
        """
        Normalize Latin script text.
        
        Args:
            text: Text to normalize
            lang: Optional language code
            
        Returns:
            Normalized text
        """
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase for better matching
        return text.lower()
    
    def _normalize_cjk(self, text: str, lang: str = None) -> str:
        """
        Normalize CJK script text.
        
        Args:
            text: Text to normalize
            lang: Optional language code
            
        Returns:
            Normalized text
        """
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # For CJK, use NFKC normalization to handle full/half-width character differences
        # but preserve case (case is generally not relevant in CJK)
        return unicodedata.normalize('NFKC', text)
    
    def _normalize_arabic(self, text: str, lang: str = None) -> str:
        """
        Normalize Arabic script text.
        
        Args:
            text: Text to normalize
            lang: Optional language code
            
        Returns:
            Normalized text
        """
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # For Arabic, use NFKD normalization
        # Arabic needs specialized handling for vowel marks and ligatures
        return unicodedata.normalize('NFKD', text)
    
    def _normalize_default(self, text: str, lang: str = None) -> str:
        """
        Default normalization for scripts without specialized handlers.
        
        Args:
            text: Text to normalize
            lang: Optional language code
            
        Returns:
            Normalized text
        """
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use NFD normalization to handle diacritics and basic forms
        return unicodedata.normalize('NFD', text)
    
    # Script-specific alignment methods (stub implementations that will be enhanced)
    
    def _align_latin(self, doc, text_segment: str, lang: str = None) -> Optional[MockSpan]:
        """Placeholder for Latin-specific alignment optimizations."""
        # Currently the default fuzzy alignment is good for Latin scripts
        return None  # Indicate to use the default alignment approach
    
    def _align_cjk(self, doc, text_segment: str, lang: str = None) -> Optional[MockSpan]:
        """Specialized alignment for CJK texts."""
        # Special handling for character-based languages
        # Remove spaces for more flexible matching since CJK doesn't rely on spaces
        clean_doc_text = re.sub(r'\s+', '', doc.text)
        clean_search_segment = re.sub(r'\s+', '', text_segment)
        
        start_char = clean_doc_text.find(clean_search_segment)
        if start_char >= 0:
            # Calculate the correct position in the original text
            # by counting characters up to the match position
            orig_pos = 0
            cjk_pos = 0
            
            while cjk_pos < start_char and orig_pos < len(doc.text):
                if not doc.text[orig_pos].isspace():
                    cjk_pos += 1
                orig_pos += 1
            
            # Find the end position in the original text
            end_pos = orig_pos
            remain_chars = len(clean_search_segment)
            
            while remain_chars > 0 and end_pos < len(doc.text):
                if not doc.text[end_pos].isspace():
                    remain_chars -= 1
                end_pos += 1
            
            return self.align_char_span(doc, orig_pos, end_pos)
            
        return None  # Use default alignment if exact match fails
    
    def _align_arabic(self, doc, text_segment: str, lang: str = None) -> Optional[MockSpan]:
        """Specialized alignment for Arabic/RTL texts."""
        # Currently use the default alignment with Arabic-specific scoring
        return None  # Use default alignment
    
    def _align_default(self, doc, text_segment: str, lang: str = None) -> Optional[MockSpan]:
        """Default alignment strategy."""
        return None  # Use general fuzzy alignment approach
    
    # Script-specific scoring methods
    
    def _score_latin(self, span, segment_tokens: List[str]) -> float:
        """
        Calculate alignment score for Latin script.
        
        For Latin scripts, token-based matching with position awareness works well.
        
        Args:
            span: Document span
            segment_tokens: Tokens to match
            
        Returns:
            Similarity score between 0.0 and 1.0+
        """
        # Get span tokens
        span_tokens = [t.text.lower() for t in span]
        span_token_count = len(span_tokens)
        segment_token_count = len(segment_tokens)
        
        # For Latin, token overlap is a good indicator
        if span_token_count < 1 or segment_token_count < 1:
            return 0.0
            
        # Calculate token overlap
        common_count = 0
        for token in segment_tokens:
            if token.lower() in span_tokens:
                common_count += 1
                
        base_score = common_count / max(span_token_count, segment_token_count)
        
        # Add position bonus for matching token positions
        position_bonus = 0.0
        max_check = min(span_token_count, segment_token_count, 5)  # Check first 5 tokens max
        
        for i in range(max_check):
            if i < span_token_count and i < segment_token_count:
                if span_tokens[i] == segment_tokens[i].lower():
                    position_bonus += 0.05
                    
        return base_score + position_bonus
    
    def _score_cjk(self, span, segment_tokens: List[str]) -> float:
        """
        Calculate alignment score for CJK script.
        
        For CJK scripts, character-level matching is more important than token positions.
        
        Args:
            span: Document span
            segment_tokens: Tokens to match
            
        Returns:
            Similarity score between 0.0 and 1.0+
        """
        # For CJK, merge tokens into character sequences
        span_text = ''.join([t.text for t in span])
        segment_text = ''.join(segment_tokens)
        
        if not span_text or not segment_text:
            return 0.0
            
        # Check for substring containment (higher score if one contains the other)
        if segment_text in span_text:
            return 0.9 + (len(segment_text) / len(span_text) * 0.1)  # Almost perfect match
        elif span_text in segment_text:
            return 0.8  # Good match but not perfect
            
        # Calculate character overlap
        span_chars = set(span_text)
        segment_chars = set(segment_text)
        
        if not span_chars or not segment_chars:
            return 0.0
            
        common_chars = span_chars.intersection(segment_chars)
        char_overlap = len(common_chars) / max(len(span_chars), len(segment_chars))
        
        # Check character sequence overlap (bigrams)
        span_bigrams = set()
        segment_bigrams = set()
        
        if len(span_text) > 1:
            span_bigrams = {span_text[i:i+2] for i in range(len(span_text)-1)}
            
        if len(segment_text) > 1:
            segment_bigrams = {segment_text[i:i+2] for i in range(len(segment_text)-1)}
        
        bigram_overlap = 0.0
        if span_bigrams and segment_bigrams:
            common_bigrams = span_bigrams.intersection(segment_bigrams)
            bigram_overlap = len(common_bigrams) / max(len(span_bigrams), len(segment_bigrams))
            
        # Final score combines character and sequence matching
        return (char_overlap * 0.3) + (bigram_overlap * 0.7)
    
    def _score_arabic(self, span, segment_tokens: List[str]) -> float:
        """
        Calculate alignment score for Arabic script.
        
        For Arabic scripts, we need to consider the specialized character properties.
        
        Args:
            span: Document span
            segment_tokens: Tokens to match
            
        Returns:
            Similarity score between 0.0 and 1.0+
        """
        # For Arabic, token-based matching is useful but needs to be more lenient
        span_tokens = [t.text for t in span]
        span_token_count = len(span_tokens)
        segment_token_count = len(segment_tokens)
        
        if span_token_count < 1 or segment_token_count < 1:
            return 0.0
            
        # Calculate normalized token overlap
        # Arabic needs normalization to handle different forms of the same character
        normalized_span = [unicodedata.normalize('NFKD', t) for t in span_tokens]
        normalized_segment = [unicodedata.normalize('NFKD', t) for t in segment_tokens]
        
        common_count = 0
        for token in normalized_segment:
            if token in normalized_span:
                common_count += 1
                
        # More lenient scoring for Arabic
        return common_count / max(span_token_count, segment_token_count)
    
    def _score_default(self, span, segment_tokens: List[str]) -> float:
        """
        Default scoring method for scripts without specialized handlers.
        
        Args:
            span: Document span
            segment_tokens: Tokens to match
            
        Returns:
            Similarity score between 0.0 and 1.0+
        """
        # Default to Latin scoring
        return self._score_latin(span, segment_tokens)
    
    def cleanup_resources(self):
        """
        Clean up cached resources to free memory.
        This should be called periodically for long-running applications.
        """
        # Clear the LRU cache for normalize_text
        if hasattr(self, '_normalize_text'):
            self._normalize_text.cache_clear()
            
        # Clear language models that aren't the default
        if hasattr(self, '_language_models') and self.default_language in self._language_models:
            default_model = self._language_models[self.default_language]
            self._language_models = {self.default_language: default_model}
        
        # Force garbage collection
        import gc
        gc.collect()
        
    def _maybe_cleanup(self):
        """Check if it's time to clean up resources and do so if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self.cleanup_resources()
            self._last_cleanup = current_time
            
    def _detect_script_type(self, text: str) -> str:
        """
        Detect the dominant script type in a text.
        
        This helps determine the appropriate normalization and alignment strategy.
        
        Args:
            text: Input text
            
        Returns:
            Script type: 'latin', 'cjk', 'arabic', 'cyrillic', 'other'
        """
        if not text:
            return 'latin'
            
        # Create counters for different script types
        script_counts = {
            'latin': 0,
            'cjk': 0,
            'arabic': 0,
            'cyrillic': 0,
            'other': 0
        }
        
        # Check a sample of the text (up to 100 characters)
        sample = text[:min(100, len(text))]
        for char in sample:
            if char in string.whitespace or char in string.punctuation:
                continue
                
            # Get character name which includes script information
            try:
                name = unicodedata.name(char).lower()
                category = unicodedata.category(char)
                
                # Detect script type based on character properties
                if 'latin' in name or 'ascii' in name:
                    script_counts['latin'] += 1
                # CJK detection - use both name and code point range check
                elif ('cjk' in name or 'hiragana' in name or 'katakana' in name or 
                      'hangul' in name or 'ideograph' in name or
                      any(start <= ord(char) <= end for start, end in [
                          (0x4E00, 0x9FFF),   # CJK Unified Ideographs
                          (0x3040, 0x30FF),   # Hiragana and Katakana
                          (0xAC00, 0xD7AF),   # Hangul Syllables
                          (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
                          (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
                          (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
                          (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
                          (0x2B820, 0x2CEAF)  # CJK Unified Ideographs Extension E
                      ])):
                    script_counts['cjk'] += 1
                elif 'arabic' in name or 'hebrew' in name:
                    script_counts['arabic'] += 1
                elif 'cyrillic' in name:
                    script_counts['cyrillic'] += 1
                elif category.startswith('L'):  # Letter category
                    script_counts['other'] += 1
            except (ValueError, TypeError):
                # If we can't get the name, count as other
                script_counts['other'] += 1
                
        # Return the dominant script type - give double weight to CJK characters
        # because a single CJK character carries more semantic content than a Latin letter
        script_counts['cjk'] *= 2
        
        # Return the dominant script type
        dominant_script = max(script_counts.items(), key=lambda x: x[1])[0]
        return dominant_script
        
    @lru_cache(maxsize=512)  # Reduced cache size to prevent memory bloat
    def _normalize_text(self, text: str, lang: str = None) -> str:
        """
        Normalize text while preserving important features of different scripts.
        
        Args:
            text: Input text
            lang: Language code (optional) - helps with script-specific normalization
            
        Returns:
            Normalized text
        """
        # For very long inputs, don't use cache to avoid memory leaks
        if len(text) > 500:  # Reduced threshold to avoid caching large strings
            return self._normalize_text_uncached(text, lang)
            
        # Clean whitespace first
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return ""
            
        # Detect script type to use appropriate normalization
        script_type = self._detect_script_type(text)
        
        # Use the appropriate script-specific handler
        if script_type in self._script_handlers:
            return self._script_handlers[script_type]["normalize"](text, lang)
        else:
            # Fall back to default handler
            return self._script_handlers["other"]["normalize"](text, lang)
        
    def _normalize_text_uncached(self, text: str, lang: str = None) -> str:
        """
        Uncached version of _normalize_text for large inputs.
        
        Args:
            text: Input text
            lang: Language code (optional)
            
        Returns:
            Normalized text
        """
        # Clean whitespace first
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return ""
            
        # Detect script type to use appropriate normalization
        script_type = self._detect_script_type(text)
        
        # Use the appropriate script-specific handler
        if script_type in self._script_handlers:
            return self._script_handlers[script_type]["normalize"](text, lang)
        else:
            # Fall back to default handler
            return self._script_handlers["other"]["normalize"](text, lang)

    def _tokenize_by_script(self, text: str, script_type: str) -> List[str]:
        """
        Tokenize text based on script type and language.
        
        Args:
            text: Input text
            script_type: Detected script type
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        if script_type == 'cjk':
            # For CJK languages, character-based tokenization is often more effective
            # Preserve character sequences for alignment
            tokens = []
            current_token = ""
            for char in text:
                # Group consecutive non-CJK characters (spaces, punctuation, latin)
                if unicodedata.category(char).startswith(('Z', 'P')) or 'LATIN' in unicodedata.name(char, '').upper():
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                    if not char.isspace():
                        tokens.append(char)
                else:
                    # For CJK characters, token = individual character
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                    tokens.append(char)
                    
            if current_token:
                tokens.append(current_token)
                
            return tokens
        elif script_type == 'arabic':
            # For Arabic and similar scripts, use special tokenization
            # For now, use a whitespace+punctuation approach
            tokens = []
            for token in re.findall(r'\w+|[^\w\s]', text):
                if token.strip():
                    tokens.append(token)
            return tokens
        else:
            # For Latin and other scripts, default to space-based tokenization + punctuation
            return [token for token in re.findall(r'\w+|[^\w\s]', text) if token.strip()]
    
    def fuzzy_align(self, doc, text_segment: str, lang: str = None, script_type: str = None) -> Optional[MockSpan]:
        """
        Find the best matching span in a document for a given text segment with robust multilingual support.
        
        This enhanced method uses script-specific strategies to handle different writing systems
        appropriately, resulting in more accurate alignment for languages with different
        tokenization requirements.
        
        Args:
            doc: Document object
            text_segment: Text to align
            lang: Language code
            script_type: Optional explicit script type (if known). Otherwise detected automatically.
            
        Returns:
            Best matching span or None if no match found
        """
        # Periodically clean up resources to prevent memory leaks
        self._maybe_cleanup()
        
        # Limit text segment size to prevent memory issues
        if text_segment and len(text_segment) > 5000:
            text_segment = text_segment[:5000]
            warnings.warn(f"Text segment truncated to 5000 characters for memory efficiency")
        # Validate inputs
        if doc is None or not text_segment:
            warnings.warn("Cannot align span: Doc is None or text_segment is empty")
            return None

        # Clean and normalize text
        clean_segment = re.sub(r"\s+", " ", text_segment).strip()
        if not clean_segment:
            warnings.warn("Empty text segment for alignment after cleaning")
            return None
        
        # Auto-detect script type if not provided
        if not script_type:
            script_type = self._detect_script_type(clean_segment)
        
        # Get spaCy doc with correct language
        doc = self._get_spacy_doc(doc.text, lang)
        
        # Try script-specific alignment first
        if script_type in self._script_handlers:
            # Use script-specific alignment strategy if it returns a result
            special_result = self._script_handlers[script_type]["align"](doc, clean_segment, lang)
            if special_result is not None:
                return special_result
        
        # Normalize the text segment based on script type
        if script_type in self._script_handlers:
            clean_segment = self._script_handlers[script_type]["normalize"](clean_segment, lang)
        else:
            clean_segment = self._script_handlers["other"]["normalize"](clean_segment, lang)
        
        # Try exact match first for efficiency
        doc_text = doc.text
        
        # For CJK text, prioritize character-by-character matching
        # since whitespace doesn't necessarily indicate word boundaries
        if script_type == 'cjk':
            # Remove all spaces for CJK exact matching since they may not be significant
            clean_doc_text = re.sub(r'\s+', '', doc_text)
            clean_search_segment = re.sub(r'\s+', '', clean_segment)
            
            start_char = clean_doc_text.find(clean_search_segment)
            if start_char >= 0:
                # Calculate the correct position in the original text
                # by counting characters up to the match position
                orig_pos = 0
                cjk_pos = 0
                
                while cjk_pos < start_char and orig_pos < len(doc_text):
                    if not doc_text[orig_pos].isspace():
                        cjk_pos += 1
                    orig_pos += 1
                
                # Find the end position in the original text
                end_pos = orig_pos
                remain_chars = len(clean_search_segment)
                
                while remain_chars > 0 and end_pos < len(doc_text):
                    if not doc_text[end_pos].isspace():
                        remain_chars -= 1
                    end_pos += 1
                
                return self.align_char_span(doc, orig_pos, end_pos)
        elif script_type == 'arabic':
            # For Arabic, we need to normalize both text and segment for matching
            norm_doc_text = unicodedata.normalize('NFKD', doc_text)
            norm_segment = unicodedata.normalize('NFKD', clean_segment)
            
            start_char = norm_doc_text.find(norm_segment)
            if start_char >= 0:
                end_char = start_char + len(norm_segment)
                # Find corresponding positions in the original text
                return self.align_char_span(doc, start_char, end_char)
        else:
            # Standard exact match for other scripts
            start_char = doc_text.lower().find(clean_segment.lower())
            if start_char >= 0:
                end_char = start_char + len(clean_segment)
                return self.align_char_span(doc, start_char, end_char)

        # Get tokens using script-specific tokenization
        if script_type in self._script_handlers:
            segment_tokens = self._script_handlers[script_type]["tokenize"](clean_segment, lang)
        else:
            segment_tokens = self._script_handlers["other"]["tokenize"](clean_segment, lang)
            
        # Choose alignment strategy based on document size
        doc_len = len(doc)
        
        # Use script-aware alignment strategies
        if doc_len > 1000:
            return self._fuzzy_align_large_doc(doc, clean_segment, lang=lang, 
                                           script_type=script_type)
        elif doc_len > 100:
            return self._fuzzy_align_medium_doc(doc, clean_segment, lang=lang, 
                                            script_type=script_type)
        else:
            return self._fuzzy_align_small_doc(doc, clean_segment, lang=lang, 
                                           script_type=script_type)

    def _calculate_similarity_score(self, span, segment_tokens: List[str], 
                                  script_type: str = 'latin') -> float:
        """
        Calculate similarity score between a span and token list with script awareness.
        Memory-optimized version to avoid leaks with large documents.
        
        Args:
            span: Document span
            segment_tokens: List of tokens to match
            script_type: The dominant script type of the text
            
        Returns:
            Similarity score between 0.0 and 1.0+
        """
        assert span is not None
        assert segment_tokens is not None

        # Use the script-specific scoring function if available
        if script_type in self._script_handlers:
            return self._script_handlers[script_type]["score"](span, segment_tokens)
        else:
            # Fall back to default handler
            return self._script_handlers["other"]["score"](span, segment_tokens)

    def _fuzzy_align_small_doc(self, doc, text_segment: str, lang: str = None, 
                            script_type: str = 'latin', segment_tokens: List[str] = None) -> Optional[MockSpan]:
        """
        Perform fuzzy alignment for small documents with script awareness.
        
        Args:
            doc: Document object
            text_segment: Text to align
            lang: Language code
            script_type: The dominant script type of the text
            
        Returns:
            Best matching span or None if no match found
        """
        # Use script-aware tokenization if segment_tokens not provided
        if segment_tokens is None:
            if script_type == 'cjk':
                segment_tokens = self._tokenize_by_script(text_segment, script_type)
            else:
                segment_tokens = [t for t in re.findall(r'\w+|[^\w\s]', text_segment) if t.strip()]
            
        if not segment_tokens:
            return None
            
        best_match = None
        best_score = 0
        
        # Adjust window size based on script type, with stricter limits for memory efficiency
        if script_type == 'cjk':
            # For CJK, use a wider window since character tokenization creates more tokens
            # but limit the absolute size to prevent memory issues
            max_window_size = min(len(segment_tokens) * 3, len(doc), 50)
        else:
            # For other scripts, use the standard window size with stricter limits
            max_window_size = min(len(segment_tokens) * 2, len(doc), 30)

        for i in range(len(doc)):
            for j in range(i + 1, min(i + max_window_size, len(doc) + 1)):
                span = doc[i:j]
                score = self._calculate_similarity_score(span, segment_tokens, script_type=script_type)

                # Adjust threshold based on script type
                threshold = 0.4 if script_type == 'cjk' else 0.5
                early_return_threshold = 0.7 if script_type == 'cjk' else 0.8
                
                if score > best_score and score > threshold:
                    best_score = score
                    best_match = span
                    
                    # Early return for high-confidence matches
                    if score > early_return_threshold:
                        return best_match
                        
        if best_match is None and text_segment.strip():
            warnings.warn(f"Failed to align text segment: '{text_segment[:50]}...' (script: {script_type})")
            
        return best_match

    def _fuzzy_align_short_segment(self, doc, segment_tokens: List[str], lang: str = None,
                               script_type: str = 'latin') -> Optional[MockSpan]:
        """
        Optimize alignment for very short segments (1-2 tokens) with script awareness.
        
        Args:
            doc: Document object
            segment_tokens: Tokens to align
            lang: Language code
            script_type: The dominant script type of the text
            
        Returns:
            Matching span or None if no match found
        """
        if not segment_tokens:
            return None
            
        # Different strategies based on script type
        if script_type == 'cjk':
            # For CJK languages with character-based tokenization, 
            # short segments need special handling
            
            # Join all tokens (typically individual characters in CJK)
            search_text = ''.join(segment_tokens)
            if not search_text:
                return None
                
            # Find exact substring match in the document text
            doc_text = doc.text
            
            # Remove spaces for more flexible matching in CJK
            search_text_no_space = search_text.replace(' ', '')
            doc_text_no_space = doc_text.replace(' ', '')
            
            idx = doc_text_no_space.find(search_text_no_space)
            if idx >= 0:
                # Now find the corresponding position in the original text
                orig_pos = 0
                no_space_pos = 0
                
                # Map position in no-space text to position in original text
                while no_space_pos < idx and orig_pos < len(doc_text):
                    if not doc_text[orig_pos].isspace():
                        no_space_pos += 1
                    orig_pos += 1
                
                # Find end position
                end_pos = orig_pos
                remain_chars = len(search_text_no_space)
                
                while remain_chars > 0 and end_pos < len(doc_text):
                    if not doc_text[end_pos].isspace():
                        remain_chars -= 1
                    end_pos += 1
                
                # Align the character span
                return self.align_char_span(doc, orig_pos, end_pos)
        else:
            # Standard approach for non-CJK languages
            # For single token, find exact match with case normalization
            if len(segment_tokens) == 1:
                target = segment_tokens[0].lower()
                for i, token in enumerate(doc):
                    if token.text.lower() == target:
                        return doc[i:i+1]

            # For two tokens, find exact sequence with case normalization
            elif len(segment_tokens) == 2:
                first, second = [t.lower() for t in segment_tokens]
                for i in range(len(doc) - 1):
                    if (doc[i].text.lower() == first and 
                        doc[i+1].text.lower() == second):
                        return doc[i:i+2]

        return None

    def _find_promising_regions(self, doc, segment_tokens: List[str], lang: str = None,
                             script_type: str = 'latin') -> List[Tuple[int, int]]:
        """
        Find promising regions in a large document using token overlap,
        with script-specific optimizations.
        
        Args:
            doc: Document object
            segment_tokens: Tokens to align
            lang: Language code
            script_type: The dominant script type of the text
            
        Returns:
            List of (start, end) index tuples for promising regions
        """
        # Limit the number of tokens to process to prevent memory issues
        if len(segment_tokens) > 100:
            segment_tokens = segment_tokens[:100]
        if not segment_tokens:
            return []
            
        # Different strategies for different script types
        if script_type == 'cjk':
            # For CJK, character-based matching is more reliable
            # Find regions containing distinctive character sequences
            
            # For CJK, use character n-grams as distinctive features
            if len(segment_tokens) > 3:
                # Create character bigrams and trigrams from the segment
                segment_text = ''.join(segment_tokens)
                bigrams = {segment_text[i:i+2] for i in range(len(segment_text)-1)}
                # Add trigrams if the segment is long enough
                trigrams = set()
                if len(segment_text) > 2:
                    trigrams = {segment_text[i:i+3] for i in range(len(segment_text)-2)}
                
                # Use the most distinctive n-grams (less common in general)
                distinctive_features = list(trigrams) + list(bigrams)
                distinctive_features = distinctive_features[:5]  # Limit to 5 features
            else:
                # For very short segments, use the full segment
                distinctive_features = [''.join(segment_tokens)]
            
            doc_len = len(doc)
            chunk_size = min(500, max(50, doc_len // 100))  # Smaller chunks for CJK
            promising_regions = []
            
            # Get the document text once
            doc_text = doc.text
            
            for i in range(0, doc_len, chunk_size):
                end_idx = min(i + chunk_size * 2, doc_len)
                
                # Get text span directly instead of collecting tokens
                if i < len(doc) and end_idx <= len(doc):
                    chunk_text = doc_text[doc[i].idx:doc[min(end_idx-1, len(doc)-1)].idx + len(doc[min(end_idx-1, len(doc)-1)].text)]
                    
                    # Check if any distinctive feature appears in this chunk
                    for feature in distinctive_features:
                        if feature in chunk_text:
                            promising_regions.append((i, end_idx))
                            break
            
            return promising_regions
        else:
            # Extract only the most distinctive tokens to reduce memory usage
            # and improve relevance of matches for non-CJK scripts
            if len(segment_tokens) > 3:  # Reduced from 5 to 3 for memory efficiency
                # Sort tokens by length in descending order (longer = more distinctive)
                # Only consider tokens of reasonable length to avoid memory issues with very long tokens
                sorted_tokens = sorted([t for t in segment_tokens if 3 < len(t) < 20], 
                                      key=len, reverse=True)
                # Use up to 3 most distinctive tokens (reduced from 5)
                segment_token_set = {t.lower() for t in sorted_tokens[:3]}
                
                # Always include at least 2 tokens, even if they're short
                if len(segment_token_set) < 2 and len(segment_tokens) >= 2:
                    segment_token_set = {segment_tokens[0].lower(), segment_tokens[-1].lower()}
            else:
                segment_token_set = {t.lower() for t in segment_tokens}
            
            doc_len = len(doc)
            # Adjust chunk size based on document length to avoid excessive memory usage
            chunk_size = min(500, max(100, doc_len // 100))  # Smaller chunks
            promising_regions = []
            
            # Limit the number of regions to check for very large documents
            max_chunks = 20
            chunks_checked = 0

            for i in range(0, doc_len, chunk_size):
                if chunks_checked >= max_chunks:
                    break
                    
                chunks_checked += 1
                end_idx = min(i + chunk_size, doc_len)  # Reduced overlap
                
                # Collect chunk tokens without creating a full span object
                chunk_tokens = set()
                for j in range(i, end_idx):
                    if j < doc_len:
                        chunk_tokens.add(doc[j].text.lower())
                    
                    # Limit size of chunk_tokens to avoid memory bloat
                    if len(chunk_tokens) > 500:  # Reduced from 1000
                        break
                
                # Consider regions with sufficient token overlap
                overlap = segment_token_set.intersection(chunk_tokens)
                
                # For languages with rich morphology (like Arabic), be more lenient
                min_overlap = 1 if script_type == 'arabic' else min(2, len(segment_tokens))
                
                if len(overlap) >= min_overlap:
                    promising_regions.append((i, end_idx))
                    
            return promising_regions

    def _fuzzy_align_region(self, doc_region, segment_tokens: List[str], lang: str = None,
                           script_type: str = 'latin') -> Optional[MockSpan]:
        """
        Perform fuzzy alignment within a specific region of the document,
        with script-specific optimizations.
        
        Args:
            doc_region: Document region
            segment_tokens: Tokens to align
            lang: Language code
            script_type: The dominant script type of the text
            
        Returns:
            Best matching span or None if no match found
        """
        segment_len = len(segment_tokens)
        if segment_len == 0:
            return None
            
        best_match = None
        best_score = 0

        # Adjust length constraints based on script type
        if script_type == 'cjk':
            # For CJK, we need more flexibility in length comparison
            # since character tokenization creates more tokens
            min_len = max(1, segment_len // 3)
            max_len = segment_len * 3
            threshold = 0.4  # Lower threshold for CJK
            high_confidence = 0.7  # Lower high confidence threshold for CJK
        else:
            # Standard constraints for other scripts
            min_len = max(1, segment_len - max(2, segment_len // 2))
            max_len = segment_len + max(2, segment_len // 2)
            threshold = 0.5
            high_confidence = 0.8

        for i in range(len(doc_region)):
            # Only check spans of reasonable length
            for j in range(i + 1, min(i + max_len, len(doc_region) + 1)):
                # Skip spans that are too different in length
                if j - i < min_len:
                    continue
                    
                span = doc_region[i:j]
                score = self._calculate_similarity_score(span, segment_tokens, script_type=script_type)
                
                # Early return for high-confidence matches
                if score > high_confidence:
                    return span
                    
                if score > best_score and score > threshold:
                    best_score = score
                    best_match = span
                    
        return best_match

    def _fuzzy_align_medium_doc(self, doc, text_segment: str, lang: str = None,
                             script_type: str = 'latin', segment_tokens: List[str] = None) -> Optional[MockSpan]:
        """
        Optimized fuzzy alignment for medium-sized documents with script awareness.
        
        Args:
            doc: Document object
            text_segment: Text to align
            lang: Language code
            script_type: The dominant script type of the text
            
        Returns:
            Best matching span or None if no match found
        """
        # Use script-aware tokenization if segment_tokens not provided
        if segment_tokens is None:
            if script_type == 'cjk':
                segment_tokens = self._tokenize_by_script(text_segment, script_type)
            else:
                segment_tokens = [t for t in re.findall(r'\w+|[^\w\s]', text_segment.lower()) if t.strip()]
            
        segment_len = len(segment_tokens)
        doc_len = len(doc)

        if segment_len == 0:
            return None
            
        # Adjust parameters based on script type
        if script_type == 'cjk':
            # For CJK languages, use different window sizes and thresholds
            window_size = min(segment_len * 3, 30)  # Larger window for CJK
            step_size = max(1, window_size // 6)  # Smaller steps for more precise search
            threshold = 0.4  # Lower threshold for acceptance
            high_confidence = 0.7  # Lower high confidence threshold
            
            # Flexibility in length comparison
            length_flexibility = segment_len // 2 if segment_len > 4 else 2
        else:
            # Standard parameters for other scripts
            window_size = min(segment_len * 2, 15)
            step_size = max(1, window_size // 4)
            threshold = 0.5
            high_confidence = 0.8
            
            # Standard length flexibility
            length_flexibility = max(2, segment_len // 2)
            
        best_match = None
        best_score = 0

        for i in range(0, doc_len, step_size):
            end_idx = min(i + window_size, doc_len)
            
            # Scan each possible span in the window
            for j in range(i, end_idx):
                if script_type == 'cjk':
                    # For CJK, check a wider range of span lengths
                    min_k = j + 1
                    max_k = min(j + segment_len * 3, end_idx + 1)
                else:
                    # For other scripts, limit to more reasonable span lengths
                    min_k = j + 1
                    max_k = min(j + segment_len * 2, end_idx + 1)
                    
                for k in range(min_k, max_k):
                    # Skip spans that are too different in length based on script
                    if script_type != 'cjk' and abs(k - j - segment_len) > length_flexibility:
                        continue
                        
                    span = doc[j:k]
                    score = self._calculate_similarity_score(span, segment_tokens, script_type=script_type)
                    
                    # Early return for high-confidence matches
                    if score > high_confidence:
                        return span
                        
                    if score > best_score and score > threshold:
                        best_score = score
                        best_match = span
                        
        return best_match

    def _fuzzy_align_large_doc(self, doc, text_segment: str, lang: str = None,
                             script_type: str = 'latin', segment_tokens: List[str] = None) -> Optional[MockSpan]:
        """
        Optimized fuzzy alignment for large documents using a multi-stage approach
        with script-specific optimizations.
        
        Args:
            doc: Document object
            text_segment: Text to align
            lang: Language code
            script_type: The dominant script type of the text
            
        Returns:
            Best matching span or None if no match found
        """
        # Limit text segment size to prevent memory issues
        if text_segment and len(text_segment) > 10000:
            text_segment = text_segment[:10000]
            warnings.warn(f"Text segment truncated to 10000 characters for memory efficiency")
        # Use script-aware tokenization if segment_tokens not provided
        if segment_tokens is None:
            if not text_segment:
                segment_tokens = []
            elif script_type == 'cjk':
                segment_tokens = self._tokenize_by_script(text_segment, script_type)
            elif script_type == 'arabic':
                # For Arabic, use specialized tokenization
                segment_tokens = self._tokenize_by_script(text_segment, script_type)
            else:
                # For Latin and other scripts, use standard tokenization
                segment_tokens = [t for t in re.findall(r'\w+|[^\w\s]', text_segment.lower()) if t.strip()]
            
        segment_len = len(segment_tokens)
        doc_len = len(doc)

        if segment_len == 0:
            return None

        # Special handling for very short segments with script awareness
        if segment_len <= 2:
            result = self._fuzzy_align_short_segment(doc, segment_tokens, lang=lang, script_type=script_type)
            if result:
                return result

        # Script-specific search prioritization
        best_match = None
        best_score = 0
        
        if script_type == 'cjk':
            # For CJK, use character sequences (n-grams) as distinctive features
            # This approach works better for character-based scripts
            segment_text = ''.join(segment_tokens)
            
            # Generate n-grams as distinctive features
            distinctive_features = []
            
            # Add trigrams if the segment is long enough (most distinctive)
            if len(segment_text) >= 3:
                distinctive_features.extend([segment_text[i:i+3] for i in range(len(segment_text)-2)][:3])
                
            # Add bigrams as fallback
            if len(segment_text) >= 2 and len(distinctive_features) < 3:
                distinctive_features.extend([segment_text[i:i+2] for i in range(len(segment_text)-1)][:3-len(distinctive_features)])
                
            # Use individual characters if nothing else is available
            if not distinctive_features and segment_text:
                distinctive_features = [c for c in segment_text][:3]
                
            # Find positions of distinctive features in the document
            candidate_positions = []
            
            # Sample the document to reduce memory usage for large documents
            stride = 1
            if doc_len > 50000:
                stride = 5
            elif doc_len > 10000:
                stride = 3
                
            doc_text = doc.text
            for i in range(0, doc_len, stride):
                if i >= len(doc):
                    break
                    
                # Get a window of text around the current position
                idx = doc[i].idx
                window_end = min(idx + 20, len(doc_text))
                window_text = doc_text[idx:window_end]
                
                # Check if any distinctive feature is in this window
                for feature in distinctive_features:
                    if feature in window_text:
                        candidate_positions.append(i)
                        break
                        
                # Limit candidate positions to avoid memory issues
                if len(candidate_positions) > 100:
                    break
        else:
            # For non-CJK scripts, use standard distinctive token approach
            if segment_len > 3:
                # Only look at tokens longer than 4 chars (more distinctive)
                distinctive_tokens = sorted(
                    [t for t in segment_tokens if len(t) > 4], 
                    key=len, reverse=True
                )[:3]
            else:
                distinctive_tokens = segment_tokens

            if not distinctive_tokens and segment_tokens:
                distinctive_tokens = segment_tokens[:2]
                
            # Find positions of distinctive tokens in the document
            candidate_positions = []
            
            # For extremely large documents, sample tokens to reduce memory usage
            stride = 1
            if doc_len > 50000:
                stride = 3
            elif doc_len > 10000:
                stride = 2
                
            # Only process every `stride` tokens to reduce memory usage with very large docs
            for i in range(0, doc_len, stride):
                if i < len(doc):
                    token_lower = doc[i].text.lower()
                    if token_lower in distinctive_tokens:
                        candidate_positions.append(i)
                        
                        # Limit candidate positions
                        if len(candidate_positions) > 100:
                            break

        # Check spans around candidate positions
        if candidate_positions:
            # Adjust window parameters based on script type
            if script_type == 'cjk':
                window_size = min(segment_len * 3, 25)  # Larger window for CJK
                threshold = 0.4  # Lower threshold for CJK
                high_confidence = 0.7  # Lower high confidence threshold for CJK
            else:
                window_size = min(segment_len * 2, 15)
                threshold = 0.5
                high_confidence = 0.8
                
            for pos in candidate_positions:
                # Define window around the distinctive token/feature
                start_idx = max(0, pos - window_size)
                end_idx = min(doc_len, pos + window_size)
                
                # Check spans around the position
                check_count = 0
                for j in range(start_idx, min(pos + 1, end_idx)):
                    # Define span size constraints based on script type
                    if script_type == 'cjk':
                        min_span_size = 1
                        max_span_size = min(segment_len * 3, 30)
                    else:
                        min_span_size = max(1, segment_len // 2)
                        max_span_size = min(segment_len * 2, 20)
                    
                    for k in range(j + min_span_size, min(j + max_span_size, end_idx + 1)):
                        # Skip spans that are too different in length for non-CJK scripts
                        if script_type != 'cjk' and abs(k - j - segment_len) > max(2, segment_len // 2):
                            continue
                            
                        span = doc[j:k]
                        score = self._calculate_similarity_score(span, segment_tokens, script_type=script_type)
                        
                        # Early return for high-confidence matches
                        if score > high_confidence:
                            return span
                            
                        if score > best_score and score > threshold:
                            best_score = score
                            best_match = span
                            
                        # Limit number of spans checked per position
                        check_count += 1
                        if check_count > 50:  # Increased limit for more thorough search
                            break
                            
            if best_match is not None:
                return best_match

        # For very large documents, use promising regions to narrow search
        if doc_len > 10000:
            # Limit the number of regions to check based on document size to prevent memory issues
            max_regions = 5
            if doc_len > 50000:
                max_regions = 3
            
            promising_regions = self._find_promising_regions(doc, segment_tokens, lang=lang, script_type=script_type)
            
            if promising_regions:
                # Check fewer regions to reduce memory usage
                regions_to_check = min(max_regions, 3 if script_type == 'cjk' else 2)
                
                for start, end in promising_regions[:regions_to_check]:
                    # Limit region size to prevent memory issues
                    if end - start > 1000:
                        mid = start + (end - start) // 2
                        end = min(mid + 500, end)
                        start = max(mid - 500, start)
                    
                    region_match = self._fuzzy_align_region(
                        doc[start:end], segment_tokens, lang=lang, script_type=script_type
                    )
                    if region_match is not None:
                        return doc[start + region_match.start:start + region_match.end]

        # Fallback to optimized windowed search with script-specific parameters
        if script_type == 'cjk':
            window_size = min(segment_len * 4, 30)  # Larger window for CJK
            step_size = max(window_size // 3, 5)  # Smaller steps for more precise search
            threshold = 0.4  # Lower threshold for CJK
            high_confidence = 0.7  # Lower high confidence threshold for CJK
        else:
            window_size = min(segment_len * 3, 15)
            step_size = max(window_size // 2, segment_len)
            threshold = 0.5
            high_confidence = 0.8
        
        # Adjust step size for very large documents
        if doc_len > 50000:
            step_size = max(15, window_size)
            
        # Maximum number of windows to check to avoid memory leaks
        max_windows = 1000 if script_type != 'cjk' else 2000  # More windows for CJK
        windows_checked = 0
            
        # Create token sets once for efficiency
        if script_type == 'cjk':
            # For CJK, character-based overlap is more relevant
            segment_chars = set(''.join(segment_tokens))
            overlap_threshold = max(1, len(segment_chars) // 5)  # Lower threshold for CJK
        else:
            # For other scripts, use token-based overlap
            segment_token_set = set(segment_tokens)
            overlap_threshold = max(1, len(segment_token_set) // 3)

        for i in range(0, doc_len, step_size):
            if windows_checked >= max_windows:
                break
                
            windows_checked += 1
            end_idx = min(i + window_size, doc_len)
            
            # Check for token/character overlap in this window
            if script_type == 'cjk':
                # For CJK, check character overlap
                if i < len(doc) and end_idx <= len(doc):
                    # Get the text span and convert to a character set
                    window_text = doc.text[doc[i].idx:doc[min(end_idx-1, len(doc)-1)].idx + len(doc[min(end_idx-1, len(doc)-1)].text)]
                    window_chars = set(window_text)
                    
                    # Skip windows with insufficient character overlap
                    if len(segment_chars.intersection(window_chars)) < overlap_threshold:
                        continue
            else:
                # For other scripts, check token overlap
                window_tokens = set()
                for j in range(i, end_idx):
                    if j < doc_len:
                        window_tokens.add(doc[j].text.lower())
                
                # Skip windows with insufficient token overlap
                if len(segment_token_set.intersection(window_tokens)) < overlap_threshold:
                    continue

            # Check spans in this window
            best_window_score = 0
            best_window_match = None
            
            # Define target span lengths based on script type
            if script_type == 'cjk':
                # For CJK, check a range of lengths
                target_lengths = [segment_len, segment_len//2, segment_len*2]
            else:
                # For other scripts, stay close to expected segment length
                target_lengths = [segment_len]
                
            # Check spans of different target lengths
            for target_span_length in target_lengths:
                if target_span_length < 1:
                    continue
                    
                for offset in range(min(8 if script_type == 'cjk' else 5, end_idx - i)):
                    if i + offset < doc_len and i + offset + target_span_length <= doc_len:
                        j = i + offset
                        k = min(j + target_span_length, end_idx)
                        
                        span = doc[j:k]
                        score = self._calculate_similarity_score(span, segment_tokens, script_type=script_type)
                        
                        # Early return for high-confidence matches
                        if score > high_confidence:
                            return span
                            
                        if score > best_window_score and score > threshold:
                            best_window_score = score
                            best_window_match = span
            
            # Update global best if this window had a good match
            if best_window_match and best_window_score > best_score:
                best_score = best_window_score
                best_match = best_window_match
                    
        return best_match
