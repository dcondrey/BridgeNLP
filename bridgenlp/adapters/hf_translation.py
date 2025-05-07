"""
Hugging Face translation adapter for BridgeNLP.

This adapter integrates text translation models from the Hugging Face Transformers
library with enhanced language detection capabilities.
"""

import threading
import time
import logging
import re
import string
import unicodedata
from typing import Dict, List, Optional, Tuple, Union, Any, Set, NamedTuple

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    pipeline = None

from ..base import BridgeBase
from ..config import BridgeConfig
from ..result import BridgeResult
from ..utils import (
    configure_device,
    get_param_with_fallback,
    get_or_create_model,
    unload_model,
    validate_text_input,
    get_model_memory_usage,
    create_model_key,
    detect_language
)

# Configure logger
logger = logging.getLogger(__name__)

# Named tuple for language detection results
class LanguageDetection(NamedTuple):
    """
    Result of language detection.
    
    Attributes:
        code: ISO language code (2-letter)
        name: Full language name (if available)
        confidence: Confidence score (0-1)
        supported: Whether this language is supported by the current translator
    """
    code: str
    name: Optional[str] = None
    confidence: float = 1.0
    supported: bool = False
    
# Named tuple for token alignment information
class TokenAlignment(NamedTuple):
    """
    Alignment between source and target tokens.
    
    Attributes:
        source_indices: Indices of source tokens
        target_indices: Indices of target tokens
        score: Confidence score for this alignment (0-1)
    """
    source_indices: List[int]
    target_indices: List[int]
    score: float = 0.0


class HuggingFaceTranslationBridge(BridgeBase):
    """
    Bridge adapter for Hugging Face translation models.
    
    This adapter integrates text translation models from the Hugging Face
    Transformers library with the BridgeNLP framework.
    """
    
    def __init__(self, 
                 model_name: str = "Helsinki-NLP/opus-mt-en-fr",
                 source_lang: Optional[str] = None,
                 target_lang: Optional[str] = None,
                 max_length: int = 100,
                 num_beams: int = 4,
                 auto_detect_language: bool = False,
                 truncation: bool = True,
                 batch_size: int = 1,
                 lazy_loading: bool = False,
                 config: Optional[BridgeConfig] = None):
        """
        Initialize the translation bridge.
        
        Args:
            model_name: Name or path of the Hugging Face model to use
            source_lang: Source language code (optional, model-specific)
            target_lang: Target language code (optional, model-specific)
            max_length: Maximum length of generated translations
            num_beams: Number of beams for beam search
            auto_detect_language: Whether to auto-detect the source language
            truncation: Whether to truncate input sequences longer than model max length
            batch_size: Batch size for batch processing (default: 1)
            lazy_loading: Whether to load the model only when first used (default: False)
            config: Optional configuration for the adapter
            
        Raises:
            ImportError: If required dependencies are not installed
        """
        # Check dependencies
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise ImportError(
                "Hugging Face Transformers not installed. "
                "Install with: pip install transformers torch"
            )
        
        # Always call the parent constructor first
        super().__init__(config)
        
        # Store model name and parameters using utility functions
        self.model_name = get_param_with_fallback(
            model_name, config, "model_name", default_value="Helsinki-NLP/opus-mt-en-fr"
        )
        
        # Determine language codes
        auto_detect = get_param_with_fallback(
            auto_detect_language, config, "params", "auto_detect_language", default_value=False
        )
        
        # Handle model name language code extraction
        extracted_source = None
        extracted_target = None
        
        # Extract language codes from model name
        # Typical format: Helsinki-NLP/opus-mt-<source_lang>-<target_lang>
        parts = self.model_name.split('/')[-1].split('-')
        if len(parts) >= 4 and parts[0] == "opus" and parts[1] == "mt":
            extracted_source = parts[2]
            extracted_target = parts[3]
        
        # Hierarchy: direct param > config param > extracted from model > default
        self.source_lang = get_param_with_fallback(
            source_lang, config, "params", "source_lang", 
            default_value=extracted_source
        )
        
        self.target_lang = get_param_with_fallback(
            target_lang, config, "params", "target_lang", 
            default_value=extracted_target
        )
        
        # Store the auto-detect flag
        self.auto_detect_language = auto_detect
        
        # Store other parameters
        self.max_length = get_param_with_fallback(
            max_length, config, "max_length", default_value=100
        )
        
        self.num_beams = get_param_with_fallback(
            num_beams, config, "params", "num_beams", default_value=4
        )
        
        self.truncation = get_param_with_fallback(
            truncation, config, "params", "truncation", default_value=True
        )
        
        self.batch_size = get_param_with_fallback(
            batch_size, config, "batch_size", default_value=1
        )
        
        self.lazy_loading = get_param_with_fallback(
            lazy_loading, config, "params", "lazy_loading", default_value=False
        )
        
        # Configure device using utility function
        self.device = configure_device(
            get_param_with_fallback(None, config, "device", default_value=-1)
        )
        
        # Create a unique key for this model in the registry
        self.model_key = create_model_key(self.model_name, "translation", self.device)
        
        # Initialize model and tokenizer if not using lazy loading
        self.translator = None
        if not self.lazy_loading:
            self._init_model()
        
        # Thread lock for model inference
        self._model_lock = threading.RLock()
        
        # Track memory usage
        self.memory_usage = 0.0
        
        # Performance metrics
        self._metrics.update({
            "total_characters": 0,
            "batch_calls": 0,
            "model_load_time": 0.0,
            "language_detections": 0,
        })
        
        # Cache detected languages for efficiency
        self._detected_langs = {}
    
    def _init_model(self):
        """Initialize the model and tokenizer."""
        if self.translator is not None:
            return
            
        start_time = time.time()
        
        try:
            # Use global model registry to share models between adapters
            def create_translator():
                return pipeline(
                    "translation", 
                    model=self.model_name, 
                    tokenizer=self.model_name,
                    device=self.device,
                    framework="pt"  # Use PyTorch
                )
            
            # Get or create the model
            self.translator = get_or_create_model(
                self.model_key,
                create_translator
            )
            
            # Track memory usage if we can
            if hasattr(self.translator, "model"):
                self.memory_usage = get_model_memory_usage(self.translator.model)
        except Exception as e:
            raise ImportError(f"Error loading model {self.model_name}: {str(e)}")
        finally:
            # Record model loading time
            with self._metrics_lock:
                self._metrics["model_load_time"] += time.time() - start_time
    
    def _tokenize_text(self, text: str, lang: str = None) -> List[str]:
        """
        Enhanced tokenization that handles multilingual text with script awareness.
        
        Args:
            text: Text to tokenize
            lang: Optional language code to improve tokenization
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # If we have a tokenizer available through the pipeline, use it
        if hasattr(self.translator, "tokenizer"):
            try:
                # Use the model's tokenizer but convert back to strings
                token_ids = self.translator.tokenizer.encode(text)
                return [self.translator.tokenizer.decode([token_id]) for token_id in token_ids
                        if token_id not in self.translator.tokenizer.all_special_ids]
            except Exception:
                pass
        
        # Detect script type for better tokenization
        script_type = self._detect_script_type(text)
        
        # Initialize aligner with script-specific tokenization handlers if available
        try:
            from ..aligner import TokenAligner
            aligner = TokenAligner()
            if hasattr(aligner, "_script_handlers") and script_type in aligner._script_handlers:
                # Use the specialized script-specific tokenization from the aligner
                return aligner._script_handlers[script_type]["tokenize"](text, lang)
        except (ImportError, AttributeError):
            # If we can't access the aligner or script handlers, fall back to direct implementation
            pass
        
        # Script-specific tokenization fallback implementation
        if script_type == 'cjk':
            # For CJK languages, use character-based tokenization
            # with grouped non-CJK characters
            tokens = []
            current_non_cjk = ""
            
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
                    # Append any accumulated non-CJK characters
                    if current_non_cjk:
                        tokens.append(current_non_cjk)
                        current_non_cjk = ""
                    # Add the CJK character as its own token
                    tokens.append(char)
                else:
                    # Accumulate non-CJK characters
                    current_non_cjk += char
                    
                    # If we hit whitespace, end the current token
                    if char.isspace() and current_non_cjk.strip():
                        tokens.append(current_non_cjk.strip())
                        current_non_cjk = " "
            
            # Add any remaining non-CJK characters
            if current_non_cjk and current_non_cjk.strip():
                tokens.append(current_non_cjk.strip())
                
            return tokens
        elif script_type == 'arabic':
            # For Arabic, use regexp-based tokenization that preserves script features
            return [t for t in re.findall(r'\w+|[^\w\s]', text) if t.strip()]
        else:
            # For Latin and most other scripts, split by whitespace
            # and then separate punctuation
            tokens = []
            for token in text.split():
                # Extract punctuation as separate tokens
                punctuation_tokens = re.findall(r'\w+|[^\w\s]', token)
                tokens.extend([t for t in punctuation_tokens if t.strip()])
            return tokens
            
    def _detect_script_type(self, text: str) -> str:
        """
        Detect the dominant script type in a text.
        
        Args:
            text: Input text
            
        Returns:
            Script type: 'latin', 'cjk', 'arabic', 'cyrillic', 'other'
        """
        if not text:
            return 'latin'
            
        # Try to use the aligner's script detection if available
        try:
            from ..aligner import TokenAligner
            aligner = TokenAligner()
            if hasattr(aligner, "_detect_script_type"):
                return aligner._detect_script_type(text)
        except (ImportError, AttributeError):
            # If we can't access the aligner, fall back to direct implementation
            pass
            
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
                elif 'cjk' in name or 'hiragana' in name or 'katakana' in name or 'hangul' in name or 'ideograph' in name:
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
                
        # Return the dominant script type
        dominant_script = max(script_counts.items(), key=lambda x: x[1])[0]
        return dominant_script
    
    def _detect_source_language(self, text: str) -> str:
        """
        Detect the source language of a text.
        
        Internal method for backward compatibility.
        
        Args:
            text: Text to detect language for
            
        Returns:
            ISO language code (2-letter)
        """
        # Use the new detect_language method but just return the code
        detection = self.detect_language(text)
        return detection.code
    
    def detect_language(self, text: str) -> LanguageDetection:
        """
        Detect the language of a text with enhanced capabilities.
        
        This method provides more detailed language detection with confidence
        scores and information about supported languages.
        
        Args:
            text: Text to detect language for
            
        Returns:
            LanguageDetection object with language code, name, confidence, and support status
            
        Raises:
            ValueError: If the input text is invalid
        """
        # Validate and clean input
        try:
            text = validate_text_input(text)
            if not text:
                raise ValueError("Cannot detect language of empty text")
        except ValueError as e:
            raise ValueError(f"Invalid input text: {str(e)}")
            
        # Check cache first using a consistent hash
        cache_key = hash(text[:min(100, len(text))])  # Use first 100 chars for efficiency
        if cache_key in self._detected_langs:
            return self._detected_langs[cache_key]
        
        # Track detection metrics
        with self._metrics_lock:
            self._metrics["language_detections"] += 1
        
        # Default fallback result
        fallback_result = LanguageDetection(
            code=self.source_lang or 'en',
            name=self._get_language_name(self.source_lang or 'en'),
            confidence=0.5,
            supported=self._is_language_supported(self.source_lang or 'en')
        )
            
        # Try different language detection libraries in order of preference
        try:
            # First attempt: try FastText-based language detection (higher accuracy)
            try:
                import fasttext
                
                # Check if we already have the model loaded
                model_key = "fasttext_language_detection"
                
                # Define model loading function
                def load_fasttext_model():
                    # FastText language identification model
                    try:
                        # Try to load from the standard location
                        return fasttext.load_model('lid.176.bin')
                    except Exception:
                        # Try to download the model if not available
                        try:
                            import os
                            import urllib.request
                            
                            model_path = os.path.join(os.path.expanduser("~"), ".cache", "fasttext", "lid.176.bin")
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            
                            if not os.path.exists(model_path):
                                url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
                                urllib.request.urlretrieve(url, model_path)
                                
                            return fasttext.load_model(model_path)
                        except Exception as e:
                            logger.warning(f"Failed to download FastText model: {e}")
                            raise
                
                # Get or load the model
                model = get_or_create_model(model_key, load_fasttext_model)
                
                # Predict with FastText
                labels, scores = model.predict(text, k=3)  # Get top 3 predictions
                
                # Extract the top language code and score
                label = labels[0].replace('__label__', '')
                score = float(scores[0])
                
                # Check if this language is supported by the translator
                is_supported = self._is_language_supported(label)
                
                # Return detailed result
                result = LanguageDetection(
                    code=label,
                    name=self._get_language_name(label),
                    confidence=score,
                    supported=is_supported
                )
                
                # Cache the result
                self._detected_langs[cache_key] = result
                return result
                
            except (ImportError, Exception) as e:
                # FastText not available or failed, try langdetect
                try:
                    import langdetect
                    from langdetect import DetectorFactory
                    
                    # Set seed for deterministic results
                    DetectorFactory.seed = 0
                    
                    # Get detailed detection results
                    detector = langdetect.detect_langs(text)
                    
                    # Extract the top language and its probability
                    top_match = detector[0]
                    lang_code = top_match.lang
                    confidence = top_match.prob  # Probability between 0 and 1
                    
                    # Check if this language is supported
                    is_supported = self._is_language_supported(lang_code)
                    
                    # Return detailed result
                    result = LanguageDetection(
                        code=lang_code,
                        name=self._get_language_name(lang_code),
                        confidence=confidence,
                        supported=is_supported
                    )
                    
                    # Cache the result
                    self._detected_langs[cache_key] = result
                    return result
                    
                except (ImportError, Exception):
                    # langdetect failed or not available, try pycld2
                    try:
                        import pycld2
                        
                        # Detect language using CLD2
                        _, _, details = pycld2.detect(text)
                        
                        # Get the top language
                        lang_code = details[0][1]
                        confidence = details[0][2] / 100.0  # Convert percentage to 0-1 scale
                        
                        # Check if this language is supported
                        is_supported = self._is_language_supported(lang_code)
                        
                        # Return detailed result
                        result = LanguageDetection(
                            code=lang_code,
                            name=self._get_language_name(lang_code),
                            confidence=confidence,
                            supported=is_supported
                        )
                        
                        # Cache the result
                        self._detected_langs[cache_key] = result
                        return result
                        
                    except (ImportError, Exception):
                        # All specialized libraries failed, fall back to basic detection
                        lang_code = detect_language(text)
                        
                        # Check if this language is supported
                        is_supported = self._is_language_supported(lang_code)
                        
                        # Return simple result with default confidence
                        result = LanguageDetection(
                            code=lang_code,
                            name=self._get_language_name(lang_code),
                            confidence=0.7,  # Default medium-high confidence
                            supported=is_supported
                        )
                        
                        # Cache the result
                        self._detected_langs[cache_key] = result
                        return result
                        
        except Exception as e:
            logger.warning(f"All language detection methods failed: {e}")
            
            # Cache the fallback result
            self._detected_langs[cache_key] = fallback_result
            return fallback_result
            
    def _is_language_supported(self, lang_code: str) -> bool:
        """
        Check if a language is supported by the current translation model.
        
        Args:
            lang_code: ISO language code
            
        Returns:
            True if the language is supported, False otherwise
        """
        # Get supported languages
        supported_langs = self.get_supported_languages()
        
        # Check if the language code is in supported source languages
        return lang_code in supported_langs.get("source_languages", [])
        
    def _get_language_name(self, lang_code: str) -> Optional[str]:
        """
        Get the full name of a language from its code.
        
        Args:
            lang_code: ISO language code
            
        Returns:
            Full language name or None if not found
        """
        # Map of ISO 639-1 language codes to full names
        LANGUAGE_NAMES = {
            "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan",
            "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek",
            "en": "English", "es": "Spanish", "et": "Estonian", "fa": "Persian", "fi": "Finnish",
            "fr": "French", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
            "hu": "Hungarian", "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
            "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian",
            "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian",
            "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
            "sk": "Slovak", "sl": "Slovenian", "sq": "Albanian", "sv": "Swedish", "ta": "Tamil",
            "te": "Telugu", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
            "vi": "Vietnamese", "zh": "Chinese"
        }
        
        return LANGUAGE_NAMES.get(lang_code)
    
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """
        Get the list of languages supported by the current translation model.
        
        This method analyzes the model name to determine supported languages.
        For Helsinki-NLP models, it extracts language codes from the model name.
        For other models, it returns a more general list based on model inspection.
        
        Returns:
            Dictionary with source_languages and target_languages lists
        """
        # Initialize default result
        result = {
            "source_languages": [],
            "target_languages": []
        }
        
        # Extract from Helsinki-NLP model naming convention
        if "Helsinki-NLP" in self.model_name or "opus-mt" in self.model_name:
            # Format is typically Helsinki-NLP/opus-mt-{src}-{tgt}
            parts = self.model_name.split('/')[-1].split('-')
            
            # Check correct pattern for Helsinki-NLP models
            if len(parts) >= 4 and parts[0] == "opus" and parts[1] == "mt":
                src_lang = parts[2]
                tgt_lang = parts[3]
                
                # Handle multi-language models
                if "+" in src_lang:
                    result["source_languages"] = src_lang.split("+")
                else:
                    result["source_languages"] = [src_lang]
                    
                if "+" in tgt_lang:
                    result["target_languages"] = tgt_lang.split("+")
                else:
                    result["target_languages"] = [tgt_lang]
                    
                return result
        
        # For mBART and other multilingual models
        if "mbart" in self.model_name.lower():
            # mBART models typically support many languages
            result["source_languages"] = [
                "ar", "cs", "de", "en", "es", "et", "fi", "fr", "gu", "hi", 
                "it", "ja", "kk", "ko", "lt", "lv", "my", "ne", "nl", "ro", 
                "ru", "si", "tr", "vi", "zh"
            ]
            result["target_languages"] = result["source_languages"].copy()
            return result
            
        # For M2M100 models
        if "m2m100" in self.model_name.lower():
            # M2M100 models support many languages
            result["source_languages"] = [
                "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", 
                "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", 
                "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", 
                "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo", 
                "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", 
                "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", 
                "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", 
                "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", 
                "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", 
                "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"
            ]
            result["target_languages"] = result["source_languages"].copy()
            return result
            
        # For NLLB models
        if "nllb" in self.model_name.lower():
            # NLLB models support even more languages
            # This is a subset of the 200+ languages supported
            result["source_languages"] = [
                "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", 
                "afr_Latn", "ajp_Arab", "aka_Latn", "amh_Ethi", "apc_Arab", 
                "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab", "asm_Beng", 
                "ast_Latn", "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", 
                "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl", "bem_Latn", 
                "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt", 
                "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn", 
                "ces_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", 
                "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt", 
                "ell_Grek", "eng_Latn", "epo_Latn", "est_Latn", "eus_Latn", 
                "ewe_Latn", "fao_Latn", "fij_Latn", "fin_Latn", "fon_Latn", 
                "fra_Latn", "fur_Latn", "fuv_Latn", "gla_Latn", "gle_Latn", 
                "glg_Latn", "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn", 
                "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", 
                "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", 
                "ita_Latn", "jav_Latn", "jpn_Jpan", "kab_Latn", "kac_Latn", 
                "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor", 
                "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khm_Khmr", "kik_Latn", 
                "kin_Latn", "kir_Cyrl", "kmb_Latn", "kmr_Latn", "knc_Arab", 
                "knc_Latn", "kon_Latn", "kor_Hang", "lao_Laoo", "lij_Latn", 
                "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn", 
                "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", 
                "lvs_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", 
                "min_Latn", "mkd_Cyrl", "mlt_Latn", "mni_Beng", "mos_Latn", 
                "mri_Latn", "mya_Mymr", "nld_Latn", "nno_Latn", "nob_Latn", 
                "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn", "oci_Latn", 
                "ory_Orya", "pag_Latn", "pan_Guru", "pap_Latn", "pbt_Arab", 
                "pes_Arab", "plt_Latn", "pol_Latn", "por_Latn", "prs_Arab", 
                "quy_Latn", "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", 
                "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr", "sin_Sinh", 
                "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", 
                "som_Latn", "sot_Latn", "spa_Latn", "srd_Latn", "srp_Cyrl", 
                "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn", "szl_Latn", 
                "tam_Taml", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn", 
                "tha_Thai", "tir_Ethi", "tpi_Latn", "tsn_Latn", "tso_Latn", 
                "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng", 
                "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", 
                "vec_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", 
                "yid_Hebr", "yor_Latn", "yue_Hant", "zho_Hans", "zho_Hant", 
                "zul_Latn"
            ]
            result["target_languages"] = result["source_languages"].copy()
            return result
            
        # For T5 models or other generic translation models
        if "t5" in self.model_name.lower() or "translation" in self.model_name.lower():
            # T5 models typically support a smaller set of languages
            result["source_languages"] = ["en", "de", "fr", "ro"]
            result["target_languages"] = ["en", "de", "fr", "ro"]
            return result
            
        # For unknown models, use explicitly set languages or fall back to defaults
        if self.source_lang:
            result["source_languages"] = [self.source_lang]
        else:
            result["source_languages"] = ["en"]  # Default fallback
            
        if self.target_lang:
            result["target_languages"] = [self.target_lang]
        else:
            result["target_languages"] = ["fr"]  # Default fallback
            
        return result
    
    def from_text(self, text: str, detect_lang: bool = False) -> BridgeResult:
        """
        Translate raw text with enhanced multilingual capabilities.
        
        Args:
            text: Raw text to translate
            detect_lang: Whether to detect language even if auto-detect is disabled
            
        Returns:
            BridgeResult containing the translation
            
        Raises:
            ValueError: If the input text is invalid
            RuntimeError: If translation fails
        """
        with self._measure_performance():
            # Validate and clean input
            try:
                text = validate_text_input(text)
                if not text:
                    return BridgeResult(tokens=[])
            except ValueError as e:
                raise ValueError(f"Invalid input text: {str(e)}")
            
            # Lazy load model if needed
            if self.translator is None:
                self._init_model()
            
            # Determine source language
            src_lang = self.source_lang
            detected_lang = None
            
            # Detect language if requested or auto-detect is enabled
            if detect_lang or (self.auto_detect_language and not src_lang):
                detected_lang = self.detect_language(text)
                src_lang = detected_lang.code
            
            # Detect script type for appropriate processing
            script_type = self._detect_script_type(text)
            
            # Generate translation with the model
            with self._model_lock:
                try:
                    translation_args = {
                        "text": text,
                        "max_length": self.max_length,
                        "truncation": self.truncation,
                        "num_beams": self.num_beams
                    }
                    
                    # Only add language parameters if they're set
                    if src_lang:
                        translation_args["src_lang"] = src_lang
                    if self.target_lang:
                        translation_args["tgt_lang"] = self.target_lang
                    
                    translation = self.translator(**translation_args)
                except Exception as e:
                    # Handle inference errors gracefully
                    raise RuntimeError(f"Translation failed: {str(e)}")
            
            # Process the translation result
            if isinstance(translation, list) and len(translation) > 0:
                translation_text = translation[0].get('translation_text', '')
            else:
                translation_text = ''
                
            # Detect script type of the translated text
            translated_script_type = self._detect_script_type(translation_text)
            
            # Tokenize the translation using script-aware tokenization
            tokens = self._tokenize_text(translation_text, lang=self.target_lang)
            
            # Preserve alignment information between original and translated text
            alignment_info = self._generate_alignment_info(text, translation_text, script_type, translated_script_type)
            
            # Build language detection information
            language_info = {}
            if detected_lang:
                language_info = {
                    "code": detected_lang.code,
                    "name": detected_lang.name,
                    "confidence": detected_lang.confidence,
                    "supported": detected_lang.supported,
                    "script_type": script_type  # Add script type information
                }
            
            # Store the translation result as a role with more metadata
            roles = [{
                "role": "TRANSLATION",
                "text": translation_text,
                "source_lang": src_lang or "unknown",
                "target_lang": self.target_lang or "unknown",
                "original_text": text,
                "original_script_type": script_type,
                "translated_script_type": translated_script_type,
                "model": self.model_name,
                "auto_detected": bool(detected_lang),
                "detection": language_info if detected_lang else None,
                "alignment": alignment_info,  # Add alignment information
                "generation_params": {
                    "max_length": self.max_length,
                    "num_beams": self.num_beams
                }
            }]
            
            # Update metrics
            with self._metrics_lock:
                self._metrics["total_tokens"] += len(tokens)
                self._metrics["total_characters"] += len(text)
            
            return BridgeResult(
                tokens=tokens,
                roles=roles
            )
    
    def detect_language_batch(self, texts: List[str]) -> List[LanguageDetection]:
        """
        Detect languages for a batch of texts efficiently.
        
        Args:
            texts: List of texts to detect languages for
            
        Returns:
            List of LanguageDetection objects
        """
        # Validate inputs
        valid_texts = []
        for text in texts:
            try:
                cleaned = validate_text_input(text)
                valid_texts.append(cleaned if cleaned else "")
            except ValueError:
                valid_texts.append("")
        
        # Track detection metrics
        with self._metrics_lock:
            self._metrics["language_detections"] += len([t for t in valid_texts if t])
        
        # Create results list
        results = []
        
        # Try to use the fastest batch-capable detector first
        try:
            # Try FastText with batch processing
            import fasttext
            
            # Check if we already have the model loaded
            model_key = "fasttext_language_detection"
            
            # Define model loading function if needed
            def load_fasttext_model():
                # FastText language identification model
                try:
                    # Try to load from the standard location
                    return fasttext.load_model('lid.176.bin')
                except Exception:
                    # Try to download the model if not available
                    try:
                        import os
                        import urllib.request
                        
                        model_path = os.path.join(os.path.expanduser("~"), ".cache", "fasttext", "lid.176.bin")
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        
                        if not os.path.exists(model_path):
                            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
                            urllib.request.urlretrieve(url, model_path)
                            
                        return fasttext.load_model(model_path)
                    except Exception as e:
                        logger.warning(f"Failed to download FastText model: {e}")
                        raise
            
            # Get or load the model
            model = get_or_create_model(model_key, load_fasttext_model)
            
            # Process texts in batches of 32 for efficiency
            batch_size = 32
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i+batch_size]
                batch = [t if t else " " for t in batch]  # Replace empty strings with space
                
                # Predict all at once
                batch_labels, batch_scores = model.predict(batch, k=1)
                
                for j, (labels, scores) in enumerate(zip(batch_labels, batch_scores)):
                    if not valid_texts[i+j]:
                        # For empty texts, add a placeholder result
                        results.append(LanguageDetection(
                            code=self.source_lang or "en",
                            name=self._get_language_name(self.source_lang or "en"),
                            confidence=0.0,
                            supported=self._is_language_supported(self.source_lang or "en")
                        ))
                        continue
                    
                    # Extract the language code and score
                    lang_code = labels[0].replace('__label__', '')
                    confidence = float(scores[0])
                    
                    # Check cache first
                    cache_key = hash(valid_texts[i+j][:min(100, len(valid_texts[i+j]))])
                    if cache_key in self._detected_langs:
                        results.append(self._detected_langs[cache_key])
                        continue
                    
                    # Create the detection result
                    detection = LanguageDetection(
                        code=lang_code,
                        name=self._get_language_name(lang_code),
                        confidence=confidence,
                        supported=self._is_language_supported(lang_code)
                    )
                    
                    # Cache the result
                    self._detected_langs[cache_key] = detection
                    results.append(detection)
            
            return results
            
        except (ImportError, Exception):
            # FastText failed, fall back to individual detection
            for text in valid_texts:
                if not text:
                    # For empty texts, add a placeholder result
                    results.append(LanguageDetection(
                        code=self.source_lang or "en",
                        name=self._get_language_name(self.source_lang or "en"),
                        confidence=0.0,
                        supported=self._is_language_supported(self.source_lang or "en")
                    ))
                else:
                    # Check cache first
                    cache_key = hash(text[:min(100, len(text))])
                    if cache_key in self._detected_langs:
                        results.append(self._detected_langs[cache_key])
                    else:
                        # Detect using the single-text method
                        detection = self.detect_language(text)
                        results.append(detection)
            
            return results
    
    def _generate_alignment_info(self, source_text: str, target_text: str, 
                               source_script_type: str, target_script_type: str) -> Dict:
        """
        Generate alignment information between source and target texts,
        with special handling for different script types.
        
        Args:
            source_text: Original text
            target_text: Translated text
            source_script_type: Script type of source text
            target_script_type: Script type of target text
            
        Returns:
            Dictionary containing alignment information and confidence scores
        """
        if not source_text or not target_text:
            return {"alignments": [], "confidence": 0.0}
            
        try:
            # Use the TokenAligner for handling alignment between original and translated text
            from ..aligner import TokenAligner
            aligner = TokenAligner()
            
            # Tokenize source and target text using script-aware tokenization
            source_tokens = self._tokenize_text(source_text, lang=None)
            target_tokens = self._tokenize_text(target_text, lang=self.target_lang)
            
            if not source_tokens or not target_tokens:
                return {"alignments": [], "confidence": 0.0}
                
            # Create simple spaCy-like docs for both texts (for alignment)
            # If these are just strings without proper documents, the alignment might fail
            # We'll return minimal alignment info with the tokens
            if not source_text.strip() or not target_text.strip():
                return {
                    "alignments": [],
                    "confidence": 0.0,
                    "source_tokens": source_tokens,
                    "target_tokens": target_tokens
                }
                
            source_doc = aligner._get_spacy_doc(source_text)
            target_doc = aligner._get_spacy_doc(target_text)
            
            # Calculate minimal alignment units based on script type
            # Different scripts require different alignment approaches
            alignments = []
            
            # Calculate the most likely alignments between segments
            # For CJK, prefer character-level alignment
            if source_script_type == "cjk" or target_script_type == "cjk":
                # For CJK, focus on character-level alignments
                # Break source text into smaller chunks based on meaning units
                chunks = []
                current_chunk = []
                
                for i, token in enumerate(source_tokens):
                    current_chunk.append(token)
                    # Use small chunks for character-based languages
                    if len(current_chunk) >= 3 or i == len(source_tokens)-1:
                        chunks.append(current_chunk)
                        current_chunk = []
                
                # For each chunk, try to find a matching span in the target
                for chunk in chunks:
                    chunk_text = ''.join(chunk)
                    span = aligner.fuzzy_align(target_doc, chunk_text, script_type=source_script_type)
                    if span:
                        # Calculate character-based positions
                        source_indices = list(range(source_tokens.index(chunk[0]), 
                                             source_tokens.index(chunk[0]) + len(chunk)))
                        
                        # Get target token indices
                        target_indices = list(range(span.start, span.end))
                        
                        # Calculate confidence
                        confidence = min(1.0, aligner._calculate_similarity_score(
                            span, chunk, script_type=source_script_type
                        ))
                        
                        alignments.append({
                            "source_indices": source_indices,
                            "target_indices": target_indices,
                            "score": confidence
                        })
            
            # For Arabic and RTL scripts, use different alignment strategy
            elif source_script_type == "arabic" or target_script_type == "arabic":
                # For Arabic, normalization is important for matching
                # Try to align meaningful phrases
                
                # Break source text into phrases
                phrases = []
                current_phrase = []
                
                for i, token in enumerate(source_tokens):
                    current_phrase.append(token)
                    # Use punctuation or position to determine phrase boundaries
                    if token in ".,;:!?" or len(current_phrase) >= 5 or i == len(source_tokens)-1:
                        phrases.append(current_phrase)
                        current_phrase = []
                
                # For each phrase, try to find a matching span in the target
                for phrase in phrases:
                    phrase_text = ' '.join(phrase)
                    # Use script-specific alignment
                    span = aligner.fuzzy_align(target_doc, phrase_text, script_type=source_script_type)
                    if span:
                        # Calculate positions
                        start_idx = source_tokens.index(phrase[0]) if phrase[0] in source_tokens else 0
                        source_indices = list(range(start_idx, start_idx + len(phrase)))
                        
                        # Get target token indices
                        target_indices = list(range(span.start, span.end))
                        
                        # Calculate confidence
                        confidence = min(1.0, aligner._calculate_similarity_score(
                            span, phrase, script_type=source_script_type
                        ))
                        
                        alignments.append({
                            "source_indices": source_indices,
                            "target_indices": target_indices,
                            "score": confidence
                        })
                        
            # For Latin and Cyrillic scripts, use token-based alignment
            else:
                # For Latin-based scripts, use word-level alignment
                # Try to align logical phrases
                
                # Break source text into phrases by punctuation
                phrases = []
                current_phrase = []
                
                for i, token in enumerate(source_tokens):
                    current_phrase.append(token)
                    # Use punctuation or position to determine phrase boundaries
                    if token in ".,;:!?" or len(current_phrase) >= 8 or i == len(source_tokens)-1:
                        if current_phrase:
                            phrases.append(current_phrase)
                            current_phrase = []
                
                # For each phrase, try to find a matching span in the target
                for phrase in phrases:
                    phrase_text = ' '.join(phrase)
                    span = aligner.fuzzy_align(target_doc, phrase_text, script_type="latin")
                    if span:
                        # Calculate positions
                        start_idx = source_tokens.index(phrase[0]) if phrase[0] in source_tokens else 0
                        source_indices = list(range(start_idx, start_idx + len(phrase)))
                        
                        # Get target token indices
                        target_indices = list(range(span.start, span.end))
                        
                        # Calculate confidence
                        confidence = min(1.0, aligner._calculate_similarity_score(
                            span, phrase, script_type="latin"
                        ))
                        
                        alignments.append({
                            "source_indices": source_indices,
                            "target_indices": target_indices,
                            "score": confidence
                        })
            
            # Calculate overall confidence score
            overall_confidence = 0.0
            if alignments:
                overall_confidence = sum(a["score"] for a in alignments) / len(alignments)
                
            return {
                "alignments": alignments,
                "confidence": overall_confidence,
                "source_tokens": source_tokens,
                "target_tokens": target_tokens
            }
            
        except (ImportError, AttributeError, Exception) as e:
            # If alignment fails, return minimal information
            logger.warning(f"Alignment generation failed: {str(e)}")
            return {"alignments": [], "confidence": 0.0}
            
    def from_batch(self, texts: List[str], detect_lang: bool = False) -> List[BridgeResult]:
        """
        Process a batch of texts with enhanced multilingual support.
        
        Args:
            texts: List of texts to process
            detect_lang: Whether to detect language even if auto-detect is disabled
            
        Returns:
            List of BridgeResult objects
        """
        with self._measure_performance():
            # Update batch calls metrics
            with self._metrics_lock:
                self._metrics["batch_calls"] += 1
            
            # Validate inputs
            valid_texts = []
            for text in texts:
                try:
                    cleaned = validate_text_input(text)
                    valid_texts.append(cleaned if cleaned else "")
                except ValueError:
                    valid_texts.append("")
            
            # Skip processing if all texts are empty
            if not any(valid_texts):
                return [BridgeResult(tokens=[]) for _ in texts]
                
            # Lazy load model if needed
            if self.translator is None:
                self._init_model()
                
            # Determine source languages and detect if needed
            source_langs = []
            detected_langs = None
            
            # Detect languages if requested or auto-detect is enabled
            if detect_lang or (self.auto_detect_language and not self.source_lang):
                detected_langs = self.detect_language_batch(valid_texts)
                source_langs = [detection.code for detection in detected_langs]
            else:
                # Use the same source language for all
                source_langs = [self.source_lang] * len(valid_texts)
                detected_langs = [None] * len(valid_texts)
                
            # Also detect script types for all texts
            script_types = [self._detect_script_type(text) if text else 'latin' for text in valid_texts]
                
            # Process texts in batches
            all_translations = []
            
            with self._model_lock:
                try:
                    for i in range(0, len(valid_texts), self.batch_size):
                        batch_texts = valid_texts[i:i+self.batch_size]
                        batch_src_langs = source_langs[i:i+self.batch_size]
                        
                        # Process each text individually since they might have different source languages
                        batch_results = []
                        for text, src_lang in zip(batch_texts, batch_src_langs):
                            if not text:
                                batch_results.append({"translation_text": ""})
                                continue
                                
                            # Create translation arguments
                            translation_args = {
                                "text": text,
                                "max_length": self.max_length,
                                "truncation": self.truncation,
                                "num_beams": self.num_beams
                            }
                            
                            # Only add language parameters if they're set
                            if src_lang:
                                translation_args["src_lang"] = src_lang
                            if self.target_lang:
                                translation_args["tgt_lang"] = self.target_lang
                            
                            # Translate
                            translation = self.translator(**translation_args)
                            
                            # Extract result
                            if isinstance(translation, list) and len(translation) > 0:
                                batch_results.append(translation[0])
                            else:
                                batch_results.append({"translation_text": ""})
                                
                        all_translations.extend(batch_results)
                        
                except Exception as e:
                    # Handle batch processing errors
                    raise RuntimeError(f"Batch translation failed: {str(e)}")
            
            # Detect script types of translated texts
            translated_script_types = []
            for translation_obj in all_translations:
                translation_text = translation_obj.get("translation_text", "")
                translated_script_types.append(
                    self._detect_script_type(translation_text) if translation_text else 'latin'
                )
            
            # Create BridgeResult objects for each translation
            results = []
            for i, (text, translation_obj, src_lang, detection, script_type, translated_script) in enumerate(
                zip(valid_texts, all_translations, source_langs, detected_langs, script_types, translated_script_types)
            ):
                translation_text = translation_obj.get("translation_text", "")
                
                # Tokenize the translation using script-aware tokenization
                tokens = self._tokenize_text(translation_text, lang=self.target_lang)
                
                # Generate alignment information with script awareness
                alignment_info = self._generate_alignment_info(
                    text, translation_text, script_type, translated_script
                )
                
                # Build language detection information if available
                language_info = {}
                if detection:
                    language_info = {
                        "code": detection.code,
                        "name": detection.name,
                        "confidence": detection.confidence,
                        "supported": detection.supported,
                        "script_type": script_type  # Add script type information
                    }
                
                # Create role information with script awareness
                roles = [{
                    "role": "TRANSLATION",
                    "text": translation_text,
                    "source_lang": src_lang or "unknown",
                    "target_lang": self.target_lang or "unknown",
                    "original_text": text,
                    "original_script_type": script_type,
                    "translated_script_type": translated_script,
                    "model": self.model_name,
                    "auto_detected": bool(detection),
                    "detection": language_info if detection else None,
                    "alignment": alignment_info,  # Add alignment information
                    "batch_index": i,
                    "generation_params": {
                        "max_length": self.max_length,
                        "num_beams": self.num_beams
                    }
                }]
                
                # Update metrics
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(tokens)
                    self._metrics["total_characters"] += len(text)
                
                results.append(BridgeResult(tokens=tokens, roles=roles))
            
            return results
    
    def from_tokens(self, tokens: List[str], detect_lang: bool = False) -> BridgeResult:
        """
        Translate pre-tokenized text.
        
        Args:
            tokens: List of pre-tokenized strings
            detect_lang: Whether to detect language even if auto-detect is disabled
            
        Returns:
            BridgeResult containing the translation
        """
        if not tokens:
            return BridgeResult(tokens=[])
            
        # Convert tokens to text and call from_text
        text = ' '.join(tokens)
        return self.from_text(text, detect_lang=detect_lang)
    
    def from_spacy(self, doc) -> "spacy.tokens.Doc":
        """
        Translate a spaCy Doc with enhanced multilingual support.
        
        Args:
            doc: spaCy Doc to translate
            
        Returns:
            The same Doc with translation information attached
        """
        if doc is None:
            raise ValueError("Input Doc cannot be None")
            
        with self._measure_performance():
            # Get the text from the Doc
            text = doc.text
            
            # Use document language if available
            lang = None
            if hasattr(doc, 'lang_'):
                lang = doc.lang_
                
            # Generate translation with language detection
            # Always detect language for spaCy docs as they may have language metadata
            result = self.from_text(text, detect_lang=True)
            
            # Add script type information to the spaCy Doc extensions
            # Register extension if it doesn't exist
            try:
                import spacy as spacy_module
                
                # Register script type extension
                if not doc.__class__.has_extension("nlp_bridge_script_type"):
                    doc.__class__.set_extension("nlp_bridge_script_type", default=None)
                
                # Register alignment extension
                if not doc.__class__.has_extension("nlp_bridge_alignment"):
                    doc.__class__.set_extension("nlp_bridge_alignment", default=None)
                
                # Set script type from our detection
                if hasattr(result, "roles") and result.roles:
                    translation_role = result.roles[0]
                    
                    # Set script type
                    if "original_script_type" in translation_role:
                        doc._.nlp_bridge_script_type = translation_role["original_script_type"]
                    
                    # Set alignment information if available
                    if "alignment" in translation_role and translation_role["alignment"]:
                        doc._.nlp_bridge_alignment = translation_role["alignment"]
            except (ImportError, AttributeError):
                # If spaCy is not available or there's another issue with extensions,
                # just continue without setting the extensions
                pass
            
            # Attach the result to the Doc
            return result.attach_to_spacy(doc)
    
    def set_languages(self, source_lang: str, target_lang: str):
        """
        Set the source and target languages for translation.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Reset language detection cache when changing source language
        if not self.auto_detect_language:
            self._detected_langs = {}
    
    def set_auto_detect(self, auto_detect: bool):
        """
        Enable or disable automatic language detection.
        
        Args:
            auto_detect: Whether to auto-detect the source language
        """
        self.auto_detect_language = auto_detect
        
        # Clear cache if disabling auto-detection
        if not auto_detect:
            self._detected_langs = {}
    
    def get_metrics(self) -> Dict[str, Union[int, float, str]]:
        """
        Get enhanced performance metrics for this adapter.
        
        Returns:
            Dictionary of metrics with additional information
        """
        # Get base metrics
        metrics = super().get_metrics()
        
        # Add memory usage information
        metrics["memory_usage_mb"] = self.memory_usage
        
        # Add efficiency metrics
        if metrics["total_tokens"] > 0 and metrics["total_characters"] > 0:
            metrics["characters_per_token"] = metrics["total_characters"] / metrics["total_tokens"]
        
        if metrics["batch_calls"] > 0:
            metrics["avg_batch_size"] = metrics["num_calls"] / metrics["batch_calls"]
            
        if metrics["model_load_time"] > 0:
            metrics["load_time_seconds"] = metrics["model_load_time"]
        
        # Add translation-specific metrics
        metrics["language_detections"] = self._metrics.get("language_detections", 0)
        metrics["auto_detect_enabled"] = self.auto_detect_language
        metrics["source_language"] = self.source_lang or "auto" if self.auto_detect_language else "unknown"
        metrics["target_language"] = self.target_lang or "unknown"
        
        # Add supported languages info
        supported = self.get_supported_languages()
        metrics["supported_source_languages"] = supported["source_languages"]
        metrics["supported_target_languages"] = supported["target_languages"]
        
        # Add supported script types
        metrics["supported_script_types"] = ["latin", "cjk", "arabic", "cyrillic", "other"]
        
        # Add detection libraries info
        detection_libraries = []
        try:
            import fasttext
            detection_libraries.append("fasttext")
        except ImportError:
            pass
            
        try:
            import langdetect
            detection_libraries.append("langdetect")
        except ImportError:
            pass
            
        try:
            import pycld2
            detection_libraries.append("pycld2")
        except ImportError:
            pass
            
        metrics["language_detection_libraries"] = detection_libraries or ["basic"]
            
        # Add model information
        metrics["model_name"] = self.model_name
        metrics["device"] = f"GPU:{self.device}" if self.device >= 0 else "CPU"
        metrics["lazy_loading"] = self.lazy_loading
        
        # Current detection cache size
        metrics["detection_cache_size"] = len(self._detected_langs)
        
        # Add multilingual capabilities information
        metrics["multilingual_capabilities"] = {
            "script_detection": True,
            "multilingual_alignment": True,
            "cjk_support": True,
            "arabic_support": True,
            "cyrillic_support": True,
            "script_aware_tokenization": True,
            "script_aware_alignment": True
        }
        
        # Add enhanced script-specific handling details
        metrics["script_handlers"] = {
            "latin": ["tokenization", "normalization", "alignment", "scoring"],
            "cjk": ["character-based tokenization", "NFKC normalization", "character alignment", "n-gram scoring"],
            "arabic": ["RTL-aware tokenization", "NFKD normalization", "phrase alignment", "normalized scoring"],
            "cyrillic": ["tokenization", "normalization", "alignment", "scoring"]
        }
        
        return metrics
    
    def cleanup(self):
        """
        Clean up resources used by this adapter.
        
        This method is called when the adapter is used as a context manager
        or when it's garbage collected.
        """
        # Unload the model if requested in config
        unload = (hasattr(self, "config") and self.config and 
                  hasattr(self.config, "unload_on_del") and self.config.unload_on_del)
                  
        if unload:
            with self._model_lock:
                # Unregister from global registry instead of just local cleanup
                unload_model(self.model_key)
                self.translator = None
                
                # Clear caches
                self._detected_langs = {}
    
    def __repr__(self) -> str:
        """
        String representation of this adapter.
        
        Returns:
            A string with adapter information
        """
        status = "loaded" if self.translator is not None else "not loaded"
        device = f"GPU:{self.device}" if self.device >= 0 else "CPU"
        src_lang = self.source_lang or "auto" if self.auto_detect_language else "unknown"
        tgt_lang = self.target_lang or "unknown"
        return f"HuggingFaceTranslationBridge(model='{self.model_name}', source={src_lang}, target={tgt_lang}, device={device}, status={status})"