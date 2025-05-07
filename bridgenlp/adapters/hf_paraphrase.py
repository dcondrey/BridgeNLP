"""
Hugging Face text paraphrasing adapter for BridgeNLP.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set

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
    create_model_key
)


class HuggingFaceParaphraseBridge(BridgeBase):
    """
    Bridge adapter for Hugging Face paraphrasing models.
    
    This adapter integrates text paraphrasing models from the Hugging Face
    Transformers library with the BridgeNLP framework.
    """
    
    def __init__(self, 
                 model_name: str = "tuner007/pegasus_paraphrase",
                 max_length: int = 100,
                 num_return_sequences: int = 1,
                 num_beams: int = 5,
                 temperature: float = 1.0,
                 do_sample: bool = True,
                 top_p: float = 0.95,
                 top_k: int = 50,
                 truncation: bool = True,
                 batch_size: int = 1,
                 lazy_loading: bool = False,
                 config: Optional[BridgeConfig] = None):
        """
        Initialize the paraphrasing bridge.
        
        Args:
            model_name: Name or path of the Hugging Face model to use
            max_length: Maximum length of generated paraphrases
            num_return_sequences: Number of paraphrase variations to generate
            num_beams: Number of beams for beam search
            temperature: Temperature for generation (higher = more diverse)
            do_sample: Whether to use sampling for generation
            top_p: Nucleus sampling parameter (higher = more diverse)
            top_k: Top-k sampling parameter (higher = more diverse)
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
            model_name, config, "model_name", default_value="tuner007/pegasus_paraphrase"
        )
        
        self.max_length = get_param_with_fallback(
            max_length, config, "max_length", default_value=100
        )
        
        # Get generation parameters
        self.num_return_sequences = get_param_with_fallback(
            num_return_sequences, config, "params", "num_return_sequences", default_value=1
        )
        
        self.num_beams = get_param_with_fallback(
            num_beams, config, "params", "num_beams", default_value=5
        )
        
        self.temperature = get_param_with_fallback(
            temperature, config, "params", "temperature", default_value=1.0
        )
        
        self.do_sample = get_param_with_fallback(
            do_sample, config, "params", "do_sample", default_value=True
        )
        
        self.top_p = get_param_with_fallback(
            top_p, config, "params", "top_p", default_value=0.95
        )
        
        self.top_k = get_param_with_fallback(
            top_k, config, "params", "top_k", default_value=50
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
        self.model_key = create_model_key(self.model_name, "paraphrase", self.device)
        
        # Initialize model and tokenizer if not using lazy loading
        self.tokenizer = None
        self.model = None
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
            "total_variants": 0,
        })
    
    def _init_model(self):
        """Initialize the model and tokenizer."""
        if self.tokenizer is not None and self.model is not None:
            return
            
        start_time = time.time()
        
        try:
            # Use global model registry to share models between adapters
            def create_models():
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                # Move model to the appropriate device
                if self.device >= 0 and torch.cuda.is_available():
                    model = model.to(f"cuda:{self.device}")
                
                return {"tokenizer": tokenizer, "model": model}
            
            # Get or create the models
            models = get_or_create_model(
                self.model_key,
                create_models
            )
            
            self.tokenizer = models["tokenizer"]
            self.model = models["model"]
            
            # Track memory usage
            self.memory_usage = get_model_memory_usage(self.model)
                
        except Exception as e:
            raise ImportError(f"Error loading model {self.model_name}: {str(e)}")
        finally:
            # Record model loading time
            with self._metrics_lock:
                self._metrics["model_load_time"] += time.time() - start_time
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Better tokenization that handles more languages and special cases.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # If we have the tokenizer available, use it for better tokenization
        if self.tokenizer:
            try:
                # Use the model's tokenizer but convert back to strings
                token_ids = self.tokenizer.encode(text)
                return [self.tokenizer.decode([token_id]) for token_id in token_ids 
                        if token_id not in self.tokenizer.all_special_ids]
            except Exception:
                pass
                
        # Fall back to simple whitespace tokenization
        return text.split()
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Generate paraphrases from raw text.
        
        Args:
            text: Raw text to paraphrase
            
        Returns:
            BridgeResult containing the paraphrases
            
        Raises:
            ValueError: If the input text is invalid
            RuntimeError: If paraphrasing fails
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
            if self.tokenizer is None or self.model is None:
                self._init_model()
            
            # Generate paraphrases with the model
            with self._model_lock:
                try:
                    # Tokenize input
                    encoding = self.tokenizer(text, padding="longest", truncation=self.truncation, return_tensors="pt")
                    
                    # Move inputs to the appropriate device
                    if self.device >= 0 and torch.cuda.is_available():
                        encoding = {k: v.to(f"cuda:{self.device}") for k, v in encoding.items()}
                    
                    # Create generation parameters
                    generation_params = {
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"],
                        "max_length": self.max_length,
                        "num_beams": self.num_beams,
                        "num_return_sequences": self.num_return_sequences,
                        "early_stopping": True
                    }
                    
                    # Only include sampling parameters if sampling is enabled
                    if self.do_sample:
                        generation_params.update({
                            "do_sample": True,
                            "temperature": self.temperature,
                        })
                        
                        # Add top_p and top_k if they're set
                        if self.top_p < 1.0:
                            generation_params["top_p"] = self.top_p
                        if self.top_k > 0:
                            generation_params["top_k"] = self.top_k
                    
                    # Generate paraphrases
                    with torch.no_grad():
                        outputs = self.model.generate(**generation_params)
                    
                    # Decode outputs
                    paraphrases = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
                    
                except Exception as e:
                    # Handle inference errors gracefully
                    raise RuntimeError(f"Paraphrasing failed: {str(e)}")
            
            # Use the first paraphrase as the main result
            main_paraphrase = paraphrases[0] if paraphrases else ""
            tokens = self._tokenize_text(main_paraphrase)
            
            # Store all paraphrases as roles with more metadata
            roles = []
            for i, p in enumerate(paraphrases):
                roles.append({
                    "role": "PARAPHRASE",
                    "text": p,
                    "variant": i+1,
                    "original_text": text,
                    "model": self.model_name,
                    "generation_params": {
                        "temperature": self.temperature if self.do_sample else 0.0,
                        "num_beams": self.num_beams,
                        "top_p": self.top_p if self.do_sample else None,
                        "top_k": self.top_k if self.do_sample else None,
                    }
                })
            
            # Update metrics
            with self._metrics_lock:
                self._metrics["total_tokens"] += len(tokens)
                self._metrics["total_characters"] += len(text)
                self._metrics["total_variants"] += len(paraphrases)
            
            return BridgeResult(
                tokens=tokens,
                roles=roles
            )
    
    def from_batch(self, texts: List[str]) -> List[BridgeResult]:
        """
        Process a batch of texts for efficient parallel processing.
        
        Args:
            texts: List of texts to process
            
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
            if self.tokenizer is None or self.model is None:
                self._init_model()
                
            # Process in batches
            all_paraphrases = []
            
            with self._model_lock:
                try:
                    for i in range(0, len(valid_texts), self.batch_size):
                        batch = valid_texts[i:i+self.batch_size]
                        
                        # Handle each text in the batch separately
                        # (we need separate results for each text)
                        batch_results = []
                        for text in batch:
                            if not text:
                                batch_results.append([])
                                continue
                            
                            # Tokenize input
                            encoding = self.tokenizer(text, padding="longest", truncation=self.truncation, return_tensors="pt")
                            
                            # Move inputs to the appropriate device
                            if self.device >= 0 and torch.cuda.is_available():
                                encoding = {k: v.to(f"cuda:{self.device}") for k, v in encoding.items()}
                            
                            # Create generation parameters
                            generation_params = {
                                "input_ids": encoding["input_ids"],
                                "attention_mask": encoding["attention_mask"],
                                "max_length": self.max_length,
                                "num_beams": self.num_beams,
                                "num_return_sequences": self.num_return_sequences,
                                "early_stopping": True
                            }
                            
                            # Only include sampling parameters if sampling is enabled
                            if self.do_sample:
                                generation_params.update({
                                    "do_sample": True,
                                    "temperature": self.temperature,
                                })
                                
                                # Add top_p and top_k if they're set
                                if self.top_p < 1.0:
                                    generation_params["top_p"] = self.top_p
                                if self.top_k > 0:
                                    generation_params["top_k"] = self.top_k
                            
                            # Generate paraphrases
                            with torch.no_grad():
                                outputs = self.model.generate(**generation_params)
                            
                            # Decode outputs
                            text_paraphrases = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
                            batch_results.append(text_paraphrases)
                            
                        all_paraphrases.extend(batch_results)
                        
                except Exception as e:
                    # Handle batch processing errors
                    raise RuntimeError(f"Batch paraphrasing failed: {str(e)}")
            
            # Create BridgeResult objects for each text
            results = []
            for i, (text, paraphrases) in enumerate(zip(valid_texts, all_paraphrases)):
                if not paraphrases:
                    results.append(BridgeResult(tokens=[]))
                    continue
                    
                # Use the first paraphrase as the main result
                main_paraphrase = paraphrases[0]
                tokens = self._tokenize_text(main_paraphrase)
                
                # Create roles for each paraphrase
                roles = []
                for j, p in enumerate(paraphrases):
                    roles.append({
                        "role": "PARAPHRASE",
                        "text": p,
                        "variant": j+1,
                        "original_text": text,
                        "model": self.model_name,
                        "batch_index": i
                    })
                
                # Update metrics
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(tokens)
                    self._metrics["total_characters"] += len(text)
                    self._metrics["total_variants"] += len(paraphrases)
                
                results.append(BridgeResult(tokens=tokens, roles=roles))
            
            return results
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Generate paraphrases from pre-tokenized text.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing the paraphrases
        """
        if not tokens:
            return BridgeResult(tokens=[])
            
        # Convert tokens to text and call from_text
        text = ' '.join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc) -> "spacy.tokens.Doc":
        """
        Generate paraphrases from a spaCy Doc.
        
        Args:
            doc: spaCy Doc to paraphrase
            
        Returns:
            The same Doc with paraphrase information attached
        """
        if doc is None:
            raise ValueError("Input Doc cannot be None")
            
        with self._measure_performance():
            # Get the text from the Doc
            text = doc.text
            
            # Generate paraphrases
            result = self.from_text(text)
            
            # Attach the result to the Doc
            return result.attach_to_spacy(doc)
    
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
        
        # Add paraphrase-specific metrics
        if metrics["total_variants"] > 0 and metrics["num_calls"] > 0:
            metrics["avg_variants_per_call"] = metrics["total_variants"] / metrics["num_calls"]
            
        # Add model information
        metrics["model_name"] = self.model_name
        metrics["device"] = f"GPU:{self.device}" if self.device >= 0 else "CPU"
        metrics["lazy_loading"] = self.lazy_loading
        
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
                self.model = None
                self.tokenizer = None
    
    def __repr__(self) -> str:
        """
        String representation of this adapter.
        
        Returns:
            A string with adapter information
        """
        status = "loaded" if self.model is not None else "not loaded"
        device = f"GPU:{self.device}" if self.device >= 0 else "CPU"
        return f"HuggingFaceParaphraseBridge(model='{self.model_name}', device={device}, status={status}, variants={self.num_return_sequences})"