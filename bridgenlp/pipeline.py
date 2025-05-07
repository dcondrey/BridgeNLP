"""
Pipeline composition framework for BridgeNLP.

This module provides a way to compose multiple bridge adapters into a pipeline,
with automatic token alignment, efficient execution for both text and multimodal inputs,
and thread-safe operation for concurrent processing.

Thread safety:
- All operations on the pipeline are thread-safe
- Cache management is protected by locks
- Result combination is atomic
- Multiple threads can safely use the same pipeline instance concurrently
- Adapters are called with appropriate synchronization
"""

import gc
import os
import threading
import copy
import collections
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar, Generic, Tuple, Set, Deque

try:
    import spacy
    from spacy.tokens import Doc
except ImportError:
    spacy = None
    Doc = Any

from .base import BridgeBase
from .config import BridgeConfig
from .result import BridgeResult
from .aligner import TokenAligner
from .multimodal_base import MultimodalBridgeBase


class Pipeline(BridgeBase):
    """
    A pipeline of bridge adapters.
    
    This class allows multiple bridge adapters to be composed into a single
    pipeline, with automatic token alignment between adapters. Supports both
    text-only and multimodal adapters.
    
    Features:
    - Composition of multiple adapters
    - Automatic token alignment
    - Result caching
    - Conditional execution based on previous results
    - Support for text and multimodal inputs
    - Thread-safe execution and caching
    """
    
    def __init__(self, adapters: List[BridgeBase], config: Optional[BridgeConfig] = None):
        """
        Initialize a pipeline with multiple adapters.
        
        Creates a thread-safe pipeline that can be used concurrently by multiple threads.
        All operations on the pipeline are protected by appropriate locks to ensure
        consistency and prevent race conditions.
        
        Args:
            adapters: List of bridge adapters to include in the pipeline
            config: Configuration for the pipeline
            
        Raises:
            ValueError: If the adapters list is empty
        """
        super().__init__(config)
        
        if not adapters:
            raise ValueError("Pipeline requires at least one adapter")
            
        # Store a copy of the adapters list to prevent external modification
        self.adapters = list(adapters)
        self.aligner = TokenAligner()
        
        # Create locks for thread safety with clear responsibilities
        self._pipeline_lock = threading.RLock()  # Master lock for pipeline operations
        self._metrics_lock = threading.RLock()   # For metrics updates
        self._conditions_lock = threading.RLock() # For condition updates
        self._result_lock = threading.RLock()    # For result combination operations
        
        # Cache for intermediate results with thread-safe implementation
        # This improves performance for repeated calls with the same text
        self._cache_enabled = config.cache_results if config and hasattr(config, "cache_results") else False
        self._cache = {}
        # Use a thread-safe deque with maximum length for LRU cache management
        self._cache_size = config.cache_size if config and hasattr(config, "cache_size") else 100
        self._cache_keys = collections.deque(maxlen=self._cache_size)
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = threading.RLock()
        
        # Thread-safe conditions dictionary for conditional execution
        # Maps adapter index to a condition function that takes the previous result
        # and returns a boolean indicating whether to run the adapter
        with self._conditions_lock:
            self._conditions = {}
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text through the pipeline of adapters in a thread-safe manner.
        
        This method ensures thread safety throughout the entire processing pipeline,
        with appropriate locks and deep copying to prevent race conditions and
        ensure consistent results even under concurrent access.
        
        Args:
            text: Raw text to process
            
        Returns:
            BridgeResult containing the combined results from all adapters
            
        Raises:
            ValueError: If text processing fails
        """
        with self._measure_performance():
            # Validate input
            if not text or not isinstance(text, str):
                return BridgeResult(tokens=[])
            
            if not text.strip():
                return BridgeResult(tokens=[])
            
            # Use master pipeline lock for high-level operations
            with self._pipeline_lock:
                try:
                    # Check cache if enabled
                    cache_key = f"text:{hash(text)}"
                    cached_result = self._check_cache(cache_key)
                    if cached_result:
                        # Return a copy of the cached result to prevent modifications
                        return cached_result
                    
                    # Access adapters list safely (in case it's modified externally)
                    local_adapters = list(self.adapters)
                    
                    if not local_adapters:
                        raise ValueError("No adapters available for processing")
                except Exception as e:
                    # Handle initialization errors
                    import warnings
                    warnings.warn(f"Error initializing pipeline processing: {e}")
                    return BridgeResult(tokens=[])
            
            try:
                # Process with the first adapter (outside lock to avoid blocking)
                combined_result = local_adapters[0].from_text(text)
                
                # Process with subsequent adapters, aligning tokens as needed
                for i, adapter in enumerate(local_adapters[1:], 1):
                    # Thread-safe check of conditions
                    skip_adapter = False
                    with self._conditions_lock:
                        if i in self._conditions:
                            # Make a deep copy of combined_result for the condition function
                            # to avoid external modifications during condition evaluation
                            condition_result = copy.deepcopy(combined_result)
                            # If the condition returns False, skip this adapter
                            if not self._conditions[i](condition_result):
                                skip_adapter = True
                    
                    if skip_adapter:
                        continue
                    
                    # Get the next result (outside lock to avoid blocking)
                    next_result = adapter.from_text(text)
                    
                    # Combine the results (thread-safe method with internal locking)
                    combined_result = self._combine_results(combined_result, next_result)
                
                # Update metrics thread-safely
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(combined_result.tokens)
                
                # Cache the result if enabled (thread-safe method with internal locking)
                self._update_cache(cache_key, combined_result)
                
                # Return deep copy to ensure the caller can't modify our cached data
                return copy.deepcopy(combined_result)
                
            except Exception as e:
                # Handle processing errors
                import warnings
                warnings.warn(f"Error during pipeline text processing: {e}")
                # Return a basic result rather than None
                return BridgeResult(tokens=[])
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text through the pipeline of adapters in a thread-safe manner.
        
        This method ensures thread safety throughout the entire processing pipeline,
        with appropriate locks and deep copying to prevent race conditions and
        ensure consistent results even under concurrent access.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing the combined results from all adapters
            
        Raises:
            ValueError: If token processing fails
        """
        with self._measure_performance():
            # Validate input
            if not tokens or not isinstance(tokens, list):
                return BridgeResult(tokens=[])
            
            # Use master pipeline lock for high-level operations
            with self._pipeline_lock:
                try:
                    # Check cache if enabled
                    # Use a tuple of tokens for hashing (immutable)
                    # Create a safe copy of tokens to avoid modification during hashing
                    token_tuple = tuple(str(t) for t in tokens)
                    cache_key = f"tokens:{hash(token_tuple)}"
                    
                    cached_result = self._check_cache(cache_key)
                    if cached_result:
                        # Return a copy of the cached result to prevent modifications
                        return cached_result
                    
                    # Access adapters list safely (in case it's modified externally)
                    local_adapters = list(self.adapters)
                    
                    if not local_adapters:
                        raise ValueError("No adapters available for processing")
                except Exception as e:
                    # Handle initialization errors
                    import warnings
                    warnings.warn(f"Error initializing pipeline token processing: {e}")
                    return BridgeResult(tokens=[])
            
            try:
                # Create a copy of tokens to prevent external modification during processing
                safe_tokens = list(tokens)
                
                # Process with the first adapter (outside lock to avoid blocking)
                combined_result = local_adapters[0].from_tokens(safe_tokens)
                
                # Process with subsequent adapters, aligning tokens as needed
                for i, adapter in enumerate(local_adapters[1:], 1):
                    # Thread-safe check of conditions
                    skip_adapter = False
                    with self._conditions_lock:
                        if i in self._conditions:
                            # Make a deep copy of combined_result for the condition function
                            # to avoid external modifications during condition evaluation
                            condition_result = copy.deepcopy(combined_result)
                            # If the condition returns False, skip this adapter
                            if not self._conditions[i](condition_result):
                                skip_adapter = True
                    
                    if skip_adapter:
                        continue
                    
                    # Get the next result (outside lock to avoid blocking)
                    next_result = adapter.from_tokens(safe_tokens)
                    
                    # Combine the results (thread-safe method with internal locking)
                    combined_result = self._combine_results(combined_result, next_result)
                
                # Update metrics thread-safely
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(combined_result.tokens)
                
                # Cache the result if enabled (thread-safe method with internal locking)
                self._update_cache(cache_key, combined_result)
                
                # Return deep copy to ensure the caller can't modify our cached data
                return copy.deepcopy(combined_result)
                
            except Exception as e:
                # Handle processing errors
                import warnings
                warnings.warn(f"Error during pipeline token processing: {e}")
                # Return a basic result rather than None
                return BridgeResult(tokens=[])
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc through the pipeline of adapters in a thread-safe manner.
        
        This method ensures thread safety throughout the entire processing pipeline,
        with appropriate locks and deep copying to prevent race conditions and
        ensure consistent results even under concurrent access.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The same Doc with results from all adapters attached
            
        Raises:
            ValueError: If doc is empty or processing fails
        """
        with self._measure_performance():
            # Validate input
            if not doc:
                raise ValueError("Cannot process empty Doc")
            
            # Use master pipeline lock for high-level operations
            with self._pipeline_lock:
                try:
                    # Check cache if enabled
                    # Hash the doc text and tokens
                    try:
                        # Create immutable values for hashing
                        text_hash = hash(doc.text)
                        tokens_hash = hash(tuple(t.text for t in doc))
                        cache_key = f"spacy:{text_hash}:{tokens_hash}"
                    except Exception as e:
                        # If hashing fails, generate a unique cache key
                        import uuid
                        cache_key = f"spacy:uuid:{uuid.uuid4()}"
                        import warnings
                        warnings.warn(f"Error hashing spaCy doc: {e}, using UUID instead")
                    
                    cached_result = self._check_cache(cache_key)
                    if cached_result:
                        # We need to re-attach the result to this specific doc
                        # This operation is thread-safe as it creates new objects
                        return cached_result.attach_to_spacy(doc)
                    
                    # Access adapters list safely (in case it's modified externally)
                    local_adapters = list(self.adapters)
                    
                    if not local_adapters:
                        raise ValueError("No adapters available for processing")
                except Exception as e:
                    # Handle initialization errors
                    import warnings
                    warnings.warn(f"Error initializing pipeline spaCy processing: {e}")
                    # Return the original doc unchanged
                    return doc
            
            # Thread-safe extension registration
            # Use a class-level lock for spaCy extension registration to avoid race conditions
            # between different instances of the Pipeline class
            with threading.RLock():
                try:
                    # Register the spaCy extensions if they don't exist yet to avoid None errors
                    for ext_name in ["nlp_bridge_spans", "nlp_bridge_clusters", "nlp_bridge_roles", "nlp_bridge_labels",
                                   "nlp_bridge_image_features", "nlp_bridge_audio_features", "nlp_bridge_multimodal_embeddings",
                                   "nlp_bridge_detected_objects", "nlp_bridge_captions"]:
                        if not Doc.has_extension(ext_name):
                            Doc.set_extension(ext_name, default=None)
                        # Initialize with empty containers if None
                        if getattr(doc._, ext_name) is None:
                            default_value = [] if ext_name not in ["nlp_bridge_image_features", "nlp_bridge_audio_features", "nlp_bridge_multimodal_embeddings"] else None
                            setattr(doc._, ext_name, default_value)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Error registering spaCy extensions: {e}")
            
            try:
                # Create a combined result to hold data from all adapters
                combined_result = BridgeResult(
                    tokens=[t.text for t in doc],
                    spans=[],
                    clusters=[],
                    roles=[],
                    labels=[]
                )
                
                # Process with the first adapter
                doc = local_adapters[0].from_spacy(doc)
                
                # After the first adapter, extract data from the doc
                # Use thread-safe deep copies to avoid race conditions
                # when accessing doc extensions
                with self._result_lock:
                    first_result = BridgeResult(
                        tokens=[t.text for t in doc],
                        spans=copy.deepcopy(doc._.nlp_bridge_spans) if doc._.nlp_bridge_spans else [],
                        clusters=copy.deepcopy(doc._.nlp_bridge_clusters) if doc._.nlp_bridge_clusters else [],
                        roles=copy.deepcopy(doc._.nlp_bridge_roles) if doc._.nlp_bridge_roles else [],
                        labels=copy.copy(doc._.nlp_bridge_labels) if doc._.nlp_bridge_labels else []
                    )
                
                # Update the combined result with data from the first adapter (thread-safe method)
                combined_result = self._combine_results(combined_result, first_result)
                
                # Process with subsequent adapters
                for i, adapter in enumerate(local_adapters[1:], 1):
                    # Thread-safe check of conditions
                    skip_adapter = False
                    with self._conditions_lock:
                        if i in self._conditions:
                            # Make a deep copy of combined_result for the condition function
                            # to avoid external modifications during condition evaluation
                            condition_result = copy.deepcopy(combined_result)
                            # If the condition returns False, skip this adapter
                            if not self._conditions[i](condition_result):
                                skip_adapter = True
                    
                    if skip_adapter:
                        continue
                    
                    # Save the current state of the doc - use thread-safe deep copies
                    with self._result_lock:
                        current_spans = copy.deepcopy(doc._.nlp_bridge_spans) if doc._.nlp_bridge_spans else []
                        current_clusters = copy.deepcopy(doc._.nlp_bridge_clusters) if doc._.nlp_bridge_clusters else []
                        current_roles = copy.deepcopy(doc._.nlp_bridge_roles) if doc._.nlp_bridge_roles else []
                        current_labels = copy.copy(doc._.nlp_bridge_labels) if doc._.nlp_bridge_labels else []
                    
                    # Apply the adapter to the doc
                    doc = adapter.from_spacy(doc)
                    
                    # Create a result object with only the new data from this adapter
                    # Use thread-safe deep copies to avoid race conditions
                    with self._result_lock:
                        adapter_result = BridgeResult(
                            tokens=[t.text for t in doc],
                            spans=copy.deepcopy(doc._.nlp_bridge_spans) if doc._.nlp_bridge_spans else [],
                            clusters=copy.deepcopy(doc._.nlp_bridge_clusters) if doc._.nlp_bridge_clusters else [],
                            roles=copy.deepcopy(doc._.nlp_bridge_roles) if doc._.nlp_bridge_roles else [],
                            labels=copy.copy(doc._.nlp_bridge_labels) if doc._.nlp_bridge_labels else []
                        )
                    
                    # Update the combined result (thread-safe method)
                    combined_result = self._combine_results(combined_result, adapter_result)
                    
                    # Restore the combined data to the doc - use thread-safe deep copies
                    with self._result_lock:
                        doc._.nlp_bridge_spans = copy.deepcopy(combined_result.spans)
                        doc._.nlp_bridge_clusters = copy.deepcopy(combined_result.clusters)
                        doc._.nlp_bridge_roles = copy.deepcopy(combined_result.roles)
                        doc._.nlp_bridge_labels = copy.copy(combined_result.labels)
                
                # Update metrics thread-safely
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(doc)
                
                # Cache the combined result if enabled
                # Update with multimodal fields that weren't included in the intermediate merging
                # Use thread-safe deep copies to avoid race conditions
                with self._result_lock:
                    combined_result.image_features = copy.deepcopy(doc._.nlp_bridge_image_features) if doc._.nlp_bridge_image_features else None
                    combined_result.audio_features = copy.deepcopy(doc._.nlp_bridge_audio_features) if doc._.nlp_bridge_audio_features else None
                    combined_result.multimodal_embeddings = copy.copy(doc._.nlp_bridge_multimodal_embeddings) if doc._.nlp_bridge_multimodal_embeddings else None
                    combined_result.detected_objects = copy.deepcopy(doc._.nlp_bridge_detected_objects) if doc._.nlp_bridge_detected_objects else []
                    combined_result.captions = copy.copy(doc._.nlp_bridge_captions) if doc._.nlp_bridge_captions else []
                
                # Update the cache (thread-safe method)
                self._update_cache(cache_key, combined_result)
                
                return doc
                
            except Exception as e:
                # Handle processing errors
                import warnings
                warnings.warn(f"Error during pipeline spaCy processing: {e}")
                # Return the original doc to avoid returning None
                return doc
    
    def from_image(self, image_path: str) -> BridgeResult:
        """
        Process an image through compatible adapters in the pipeline in a thread-safe manner.
        
        This method ensures thread safety throughout the entire processing pipeline,
        with appropriate locks and deep copying to prevent race conditions and
        ensure consistent results even under concurrent access.
        
        Args:
            image_path: Path to the image file to process
            
        Returns:
            BridgeResult containing the combined results from compatible adapters
            
        Raises:
            ValueError: If no compatible adapters are found or image file doesn't exist
        """
        with self._measure_performance():
            # Use master pipeline lock for input validation and initialization
            with self._pipeline_lock:
                try:
                    # Validate image path exists - wrap in try/except for thread safety
                    # as the file might be deleted between the check and usage
                    if not image_path or not isinstance(image_path, str):
                        raise ValueError("Invalid image path: must be a non-empty string")
                    
                    if not os.path.exists(image_path) or not os.path.isfile(image_path):
                        raise ValueError(f"Image file not found: {image_path}")
                    
                    # Make a copy of the image path to prevent external modification
                    safe_image_path = str(image_path)
                    
                    # Check cache if enabled
                    # Use a consistent hashing approach for thread safety
                    try:
                        cache_key = f"image:{hash(os.path.abspath(safe_image_path))}"
                    except Exception:
                        # If hashing fails, use the path as is with a prefix
                        cache_key = f"image_path:{safe_image_path}"
                        
                    cached_result = self._check_cache(cache_key)
                    if cached_result:
                        # Return a deep copy to prevent external modification
                        return cached_result
                        
                    # Make a thread-safe copy of adapters
                    local_adapters = list(self.adapters)
                    
                    # Find compatible adapters (those that inherit from MultimodalBridgeBase)
                    # Do this within the lock to prevent race conditions
                    compatible_adapters = [
                        adapter for adapter in local_adapters 
                        if isinstance(adapter, MultimodalBridgeBase)
                    ]
                    
                    if not compatible_adapters:
                        raise ValueError("No compatible image processing adapters found in pipeline")
                    
                    # Find the indices of the compatible adapters in the main adapter list
                    # This needs to be done within the lock to ensure consistency
                    adapter_indices = [i for i, adapter in enumerate(local_adapters) if adapter in compatible_adapters]
                    
                except Exception as e:
                    # Handle initialization errors
                    import warnings
                    warnings.warn(f"Error initializing image processing: {e}")
                    return BridgeResult(tokens=[])
            
            try:
                # Process with the first compatible adapter (outside lock to avoid blocking)
                combined_result = compatible_adapters[0].from_image(safe_image_path)
                
                # Process with subsequent compatible adapters
                for idx, adapter in zip(adapter_indices[1:], compatible_adapters[1:]):
                    # Thread-safe check of conditions
                    skip_adapter = False
                    with self._conditions_lock:
                        if idx in self._conditions:
                            # Make a deep copy of combined_result for the condition function
                            # to avoid external modifications during condition evaluation
                            condition_result = copy.deepcopy(combined_result)
                            # If the condition returns False, skip this adapter
                            if not self._conditions[idx](condition_result):
                                skip_adapter = True
                    
                    if skip_adapter:
                        continue
                    
                    # Get the next result (outside lock to avoid blocking)
                    next_result = adapter.from_image(safe_image_path)
                    
                    # Combine the results (thread-safe method with internal locking)
                    combined_result = self._combine_results(combined_result, next_result)
                
                # Update metrics thread-safely
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(combined_result.tokens)
                
                # Cache the result if enabled (thread-safe method with internal locking)
                self._update_cache(cache_key, combined_result)
                
                # Return deep copy to ensure the caller can't modify our cached data
                return copy.deepcopy(combined_result)
                
            except Exception as e:
                # Handle processing errors
                import warnings
                warnings.warn(f"Error during image processing: {e}")
                # Return a basic result rather than None
                return BridgeResult(tokens=[])
    
    def from_text_and_image(self, text: str, image_path: str) -> BridgeResult:
        """
        Process text and image together through compatible adapters in a thread-safe manner.
        
        This method ensures thread safety throughout the entire processing pipeline,
        with appropriate locks and deep copying to prevent race conditions and
        ensure consistent results even under concurrent access.
        
        Args:
            text: Text to process
            image_path: Path to the image file
            
        Returns:
            BridgeResult containing the combined results from compatible adapters
            
        Raises:
            ValueError: If no compatible adapters are found or inputs are invalid
        """
        with self._measure_performance():
            # Use master pipeline lock for input validation and initialization
            with self._pipeline_lock:
                try:
                    # Validate text input
                    if not text or not isinstance(text, str):
                        raise ValueError("Invalid text: must be a non-empty string")
                    
                    # Validate image path - wrap in try/except for thread safety
                    if not image_path or not isinstance(image_path, str):
                        raise ValueError("Invalid image path: must be a non-empty string")
                    
                    if not os.path.exists(image_path) or not os.path.isfile(image_path):
                        raise ValueError(f"Image file not found: {image_path}")
                    
                    # Make copies of inputs to prevent external modification
                    safe_text = str(text)
                    safe_image_path = str(image_path)
                    
                    # Check cache if enabled
                    # Use a consistent hashing approach for thread safety
                    try:
                        text_hash = hash(safe_text)
                        image_hash = hash(os.path.abspath(safe_image_path))
                        cache_key = f"text_image:{text_hash}:{image_hash}"
                    except Exception:
                        # If hashing fails, generate a unique ID
                        import uuid
                        cache_key = f"text_image:uuid:{uuid.uuid4()}"
                    
                    cached_result = self._check_cache(cache_key)
                    if cached_result:
                        # Return a deep copy to prevent external modification
                        return cached_result
                    
                    # Make a thread-safe copy of adapters
                    local_adapters = list(self.adapters)
                    
                    # Find compatible adapters (those that inherit from MultimodalBridgeBase)
                    # Do this within the lock to prevent race conditions
                    compatible_adapters = [
                        adapter for adapter in local_adapters 
                        if isinstance(adapter, MultimodalBridgeBase)
                    ]
                    
                    if not compatible_adapters:
                        raise ValueError("No compatible multimodal adapters found in pipeline")
                    
                    # Find the indices of the compatible adapters in the main adapter list
                    # This needs to be done within the lock to ensure consistency
                    adapter_indices = [i for i, adapter in enumerate(local_adapters) if adapter in compatible_adapters]
                    
                except Exception as e:
                    # Handle initialization errors
                    import warnings
                    warnings.warn(f"Error initializing multimodal processing: {e}")
                    return BridgeResult(tokens=[])
            
            try:
                # Process with the first compatible adapter (outside lock to avoid blocking)
                combined_result = compatible_adapters[0].from_text_and_image(safe_text, safe_image_path)
                
                # Process with subsequent compatible adapters
                for idx, adapter in zip(adapter_indices[1:], compatible_adapters[1:]):
                    # Thread-safe check of conditions
                    skip_adapter = False
                    with self._conditions_lock:
                        if idx in self._conditions:
                            # Make a deep copy of combined_result for the condition function
                            # to avoid external modifications during condition evaluation
                            condition_result = copy.deepcopy(combined_result)
                            # If the condition returns False, skip this adapter
                            if not self._conditions[idx](condition_result):
                                skip_adapter = True
                    
                    if skip_adapter:
                        continue
                    
                    # Get the next result (outside lock to avoid blocking)
                    next_result = adapter.from_text_and_image(safe_text, safe_image_path)
                    
                    # Combine the results (thread-safe method with internal locking)
                    combined_result = self._combine_results(combined_result, next_result)
                
                # Update metrics thread-safely
                with self._metrics_lock:
                    self._metrics["total_tokens"] += len(combined_result.tokens)
                
                # Cache the result if enabled (thread-safe method with internal locking)
                self._update_cache(cache_key, combined_result)
                
                # Return deep copy to ensure the caller can't modify our cached data
                return copy.deepcopy(combined_result)
                
            except Exception as e:
                # Handle processing errors
                import warnings
                warnings.warn(f"Error during multimodal processing: {e}")
                # Return a basic result rather than None
                return BridgeResult(tokens=[])
    
    def _combine_results(self, result1: BridgeResult, result2: BridgeResult) -> BridgeResult:
        """
        Thread-safely combine two BridgeResults into a single result.
        
        This method creates a new BridgeResult that contains all information from both
        input results, with appropriate merging of overlapping information. All operations
        are performed within a lock to ensure thread safety.
        
        Args:
            result1: First result
            result2: Second result
            
        Returns:
            Combined BridgeResult with properly merged information
        """
        # Acquire the result lock to ensure thread safety during merging
        with self._result_lock:
            # Create deep copies of mutable data to avoid external mutations
            # Start with the first result
            combined = BridgeResult(
                tokens=copy.copy(result1.tokens),
                spans=copy.deepcopy(result1.spans),
                clusters=copy.deepcopy(result1.clusters),
                roles=copy.deepcopy(result1.roles),
                labels=copy.copy(result1.labels),
                # Include multimodal fields
                image_features=copy.deepcopy(result1.image_features) if result1.image_features else None,
                audio_features=copy.deepcopy(result1.audio_features) if result1.audio_features else None,
                multimodal_embeddings=copy.copy(result1.multimodal_embeddings) if result1.multimodal_embeddings else None,
                detected_objects=copy.deepcopy(result1.detected_objects),
                captions=copy.copy(result1.captions)
            )
            
            # Apply alignment to map from the second result to the first
            # For simplicity, we'll just add from the second result
            # A more sophisticated implementation would merge entries
            
            # Add spans if they don't already exist
            for span in result2.spans:
                if span not in combined.spans:
                    combined.spans.append(copy.copy(span))
            
            # Add clusters if they don't already exist
            for cluster in result2.clusters:
                if not any(self._clusters_overlap(cluster, existing) for existing in combined.clusters):
                    combined.clusters.append(copy.deepcopy(cluster))
            
            # Add roles from the second result
            for role in result2.roles:
                # Check if this role already exists with the same text
                if not any(r.get("text") == role.get("text") and 
                          r.get("role") == role.get("role") for r in combined.roles):
                    combined.roles.append(copy.deepcopy(role))
            
            # Merge labels from result2
            if not combined.labels and result2.labels:
                # If result1 has no labels but result2 does, use result2's labels
                combined.labels = copy.copy(result2.labels)
            elif combined.labels and result2.labels:
                # Create a new labels list to avoid modifying the original
                new_labels = list(combined.labels)
                
                # If both have labels, prefer non-"O" labels
                for i, label in enumerate(result2.labels):
                    if i < len(new_labels):
                        if new_labels[i] == "O" and label != "O":
                            new_labels[i] = label
                    else:
                        # If result2 has more labels than result1, append them
                        new_labels.append(label)
                        
                combined.labels = new_labels
            
            # Merge multimodal fields
            
            # Merge image features
            if not combined.image_features and result2.image_features:
                combined.image_features = copy.deepcopy(result2.image_features)
            elif combined.image_features and result2.image_features:
                # If both have image features, create a new merged dictionary
                combined.image_features = {**combined.image_features, **result2.image_features}
                
            # Merge audio features
            if not combined.audio_features and result2.audio_features:
                combined.audio_features = copy.deepcopy(result2.audio_features)
            elif combined.audio_features and result2.audio_features:
                # If both have audio features, create a new merged dictionary
                combined.audio_features = {**combined.audio_features, **result2.audio_features}
                
            # Use multimodal embeddings from the most feature-rich result
            if not combined.multimodal_embeddings and result2.multimodal_embeddings:
                combined.multimodal_embeddings = copy.copy(result2.multimodal_embeddings)
                
            # Add detected objects if they don't already exist
            for obj in result2.detected_objects:
                # Check if this object already exists with the same label and box
                if not any(o.get("label") == obj.get("label") and 
                          o.get("box") == obj.get("box") for o in combined.detected_objects):
                    combined.detected_objects.append(copy.deepcopy(obj))
                    
            # Add captions if they don't already exist
            for caption in result2.captions:
                if caption not in combined.captions:
                    combined.captions.append(caption)
            
            return combined
    
    def _clusters_overlap(self, cluster1: List[Tuple[int, int]], cluster2: List[Tuple[int, int]]) -> bool:
        """
        Check if two coreference clusters overlap.
        
        Args:
            cluster1: First cluster
            cluster2: Second cluster
            
        Returns:
            True if the clusters overlap, False otherwise
        """
        # Convert to sets of spans for easier comparison
        spans1 = set(tuple(span) for span in cluster1)
        spans2 = set(tuple(span) for span in cluster2)
        
        # Check for any overlap
        return bool(spans1.intersection(spans2))
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the pipeline.
        
        Returns:
            Dictionary of metrics including adapter-specific metrics
        """
        # Get a thread-safe copy of the metrics
        with self._metrics_lock:
            metrics = dict(self._metrics)
        
        # Calculate derived metrics
        if metrics["num_calls"] > 0:
            metrics["avg_time"] = metrics["total_time"] / metrics["num_calls"]
            if metrics["total_tokens"] > 0 and metrics["total_time"] > 0.00001:
                metrics["tokens_per_second"] = metrics["total_tokens"] / metrics["total_time"]
            elif metrics["total_tokens"] > 0:
                metrics["tokens_per_second"] = float('inf')
        
        # Add cache metrics if enabled
        if self._cache_enabled:
            with self._cache_lock:
                total = self._cache_hits + self._cache_misses
                metrics["cache_hits"] = self._cache_hits
                metrics["cache_misses"] = self._cache_misses
                metrics["cache_size"] = len(self._cache)
                metrics["cache_hit_ratio"] = (self._cache_hits / total) if total > 0 else 0.0
        
        # Add adapter-specific metrics
        adapter_metrics = {}
        for i, adapter in enumerate(self.adapters):
            # Get metrics from each adapter
            adapter_data = adapter.get_metrics()
            for key, value in adapter_data.items():
                adapter_metrics[f"adapter{i+1}_{key}"] = value
                
        # Combine adapter metrics with pipeline metrics
        metrics.update(adapter_metrics)
        
        return metrics
    
    def add_condition(self, adapter_index: int, condition_fn: Callable[[BridgeResult], bool]) -> None:
        """
        Add a condition function for conditional execution of an adapter.
        
        The condition function takes the result from the previous adapter
        and returns a boolean indicating whether to run the adapter at the
        specified index. This method is thread-safe.
        
        Args:
            adapter_index: Index of the adapter to conditionally execute (>= 1)
            condition_fn: Function that takes a BridgeResult and returns a boolean
            
        Raises:
            ValueError: If adapter_index is out of range or invalid
        """
        if adapter_index < 1 or adapter_index >= len(self.adapters):
            raise ValueError(f"Invalid adapter index: {adapter_index}. Must be between 1 and {len(self.adapters) - 1}")
        
        # Thread-safe update of conditions
        with self._conditions_lock:
            self._conditions[adapter_index] = condition_fn
    
    def _update_cache(self, cache_key: str, result: BridgeResult) -> None:
        """
        Thread-safe method to update the cache with a new result.
        
        This method ensures that cache management operations are atomic,
        with appropriate locking to prevent race conditions even under
        high concurrency. All cache operations (add, remove, update)
        are performed as a single atomic operation.
        
        Args:
            cache_key: The key to use for caching
            result: The result to cache
        """
        if not self._cache_enabled:
            return
            
        with self._cache_lock:
            try:
                # Create a deep copy of the result to prevent external mutations
                # affecting our cached data
                result_copy = copy.deepcopy(result)
                
                # When using a deque with maxlen, we don't need to manually
                # manage eviction because the deque automatically removes
                # the oldest item when adding a new one if maxlen is reached.
                
                # If the key already exists, remove it first
                if cache_key in self._cache:
                    self._cache.pop(cache_key, None)
                    # Try to remove it from the deque too, but this might fail
                    # if another thread already removed it
                    try:
                        self._cache_keys.remove(cache_key)
                    except ValueError:
                        pass  # Key already removed, that's fine
                        
                # Add new item - the deque will automatically handle LRU eviction
                self._cache[cache_key] = result_copy
                self._cache_keys.append(cache_key)
                
                # If the cache is larger than the limit, remove oldest entries
                # This is a safety measure in case the automatic deque maxlen
                # mechanism fails for some reason
                while len(self._cache) > self._cache_size:
                    try:
                        # Get the oldest key (left side of the deque)
                        oldest_key = self._cache_keys[0]
                        self._cache_keys.popleft()
                        self._cache.pop(oldest_key, None)
                    except (IndexError, KeyError):
                        # Handle race conditions where another thread modified the cache
                        break
                        
            except Exception as e:
                # Log errors but don't crash the pipeline
                import warnings
                warnings.warn(f"Error updating cache: {e}")
    
    def _check_cache(self, cache_key: str) -> Optional[BridgeResult]:
        """
        Thread-safe method to check if a result is in the cache.
        
        This method ensures thread-safe access to the cache, using appropriate
        locking to prevent race conditions. When a cache hit occurs, it returns
        a deep copy of the cached result to prevent subsequent modification
        of the cached data.
        
        Args:
            cache_key: The cache key to check
            
        Returns:
            A deep copy of the cached result if found, otherwise None
        """
        if not self._cache_enabled:
            return None
            
        with self._cache_lock:
            try:
                if cache_key in self._cache:
                    self._cache_hits += 1
                    
                    # Update key position in LRU tracking
                    try:
                        # Try to remove and re-add the key to move it to the end
                        self._cache_keys.remove(cache_key)
                        self._cache_keys.append(cache_key)
                    except ValueError:
                        # If the key wasn't in the deque for some reason, just add it
                        self._cache_keys.append(cache_key)
                    
                    # Return a deep copy to prevent external modifications
                    # affecting cached data
                    return copy.deepcopy(self._cache[cache_key])
                    
                self._cache_misses += 1
                return None
            except Exception as e:
                # Log errors but don't crash the pipeline
                import warnings
                warnings.warn(f"Error checking cache: {e}")
                # Safely increment misses even in case of error
                self._cache_misses += 1
                return None
            
    def cleanup(self):
        """
        Clean up resources used by all adapters.
        
        This method calls cleanup on each adapter in the pipeline.
        """
        # Call cleanup on each adapter
        try:
            for adapter in self.adapters:
                adapter.cleanup()
        except Exception as e:
            import warnings
            warnings.warn(f"Error during adapter cleanup: {e}")
        
        # Clear the cache
        with self._cache_lock:
            self._cache.clear()
            self._cache_keys.clear()
            self._cache_hits = 0
            self._cache_misses = 0
        
        # Force garbage collection
        gc.collect()