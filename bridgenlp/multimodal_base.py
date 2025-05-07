"""
Base abstract classes for BridgeNLP multimodal adapters.
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO

try:
    import spacy
    from spacy.tokens import Doc
except ImportError:
    # Provide a helpful error message but allow the module to be imported
    print("Warning: spaCy not installed. Install with: pip install spacy")
    spacy = None
    Doc = Any

try:
    import numpy as np
except ImportError:
    # Provide a helpful error message but allow the module to be imported
    print("Warning: NumPy not installed. Install with: pip install numpy")
    np = None

from .base import BridgeBase
from .config import BridgeConfig
from .result import BridgeResult


class MultimodalBridgeBase(BridgeBase):
    """
    Abstract base class for multimodal bridge adapters.
    
    Extends the text-based BridgeBase to support multimodal inputs
    including images, audio, and mixed content.
    """
    
    def __init__(self, config: Optional[BridgeConfig] = None) -> None:
        """
        Initialize the multimodal bridge adapter with optional configuration.
        
        Args:
            config: Configuration for the adapter
        """
        super().__init__(config)
        self._model_loaded = False
    
    @abstractmethod
    def from_image(self, image_path: str) -> BridgeResult:
        """
        Process an image and return structured results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            BridgeResult containing the processed information
        """
        pass
    
    def from_image_batch(self, image_paths: List[str]) -> List[BridgeResult]:
        """
        Process a batch of images for efficient processing.
        
        This method can be overridden by adapters that support batch processing.
        The default implementation calls from_image for each item, but tracks
        total time and tokens for more accurate metrics.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of BridgeResult objects
        """
        with self._measure_performance():
            results = []
            for path in image_paths:
                # Don't re-measure performance for individual items
                # since we're already measuring for the whole batch
                try:
                    result = self.from_image(path)
                    results.append(result)
                    
                    # Track tokens for metrics (but don't double-count time)
                    if hasattr(self, "_metrics") and "total_tokens" in self._metrics:
                        with self._metrics_lock:
                            self._metrics["total_tokens"] += len(result.tokens)
                except Exception as e:
                    # Create an error result and continue processing the batch
                    error_result = BridgeResult(
                        tokens=["error"],
                        labels=[f"Error processing {path}: {str(e)}"]
                    )
                    results.append(error_result)
                    
                    # Track the error in metrics
                    if hasattr(self, "_metrics") and "errors" in self._metrics:
                        with self._metrics_lock:
                            self._metrics["errors"] += 1
            
            # Correct the call count (batch = 1 call, not len(image_paths) calls)
            if hasattr(self, "_metrics") and "num_calls" in self._metrics:
                with self._metrics_lock:
                    # Subtract len(image_paths)-1 since we added 1 in _measure_performance
                    self._metrics["num_calls"] -= (len(image_paths) - 1)
            
            return results
    
    @abstractmethod
    def from_audio(self, audio_path: str) -> BridgeResult:
        """
        Process an audio file and return structured results.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            BridgeResult containing the processed information
        """
        pass
    
    def from_audio_batch(self, audio_paths: List[str]) -> List[BridgeResult]:
        """
        Process a batch of audio files for efficient processing.
        
        This method can be overridden by adapters that support batch processing.
        The default implementation calls from_audio for each item, but tracks
        total time and tokens for more accurate metrics.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of BridgeResult objects
        """
        with self._measure_performance():
            results = []
            for path in audio_paths:
                # Don't re-measure performance for individual items
                # since we're already measuring for the whole batch
                try:
                    result = self.from_audio(path)
                    results.append(result)
                    
                    # Track tokens for metrics (but don't double-count time)
                    if hasattr(self, "_metrics") and "total_tokens" in self._metrics:
                        with self._metrics_lock:
                            self._metrics["total_tokens"] += len(result.tokens)
                except Exception as e:
                    # Create an error result and continue processing the batch
                    error_result = BridgeResult(
                        tokens=["error"],
                        labels=[f"Error processing {path}: {str(e)}"]
                    )
                    results.append(error_result)
                    
                    # Track the error in metrics
                    if hasattr(self, "_metrics") and "errors" in self._metrics:
                        with self._metrics_lock:
                            self._metrics["errors"] += 1
            
            # Correct the call count (batch = 1 call, not len(audio_paths) calls)
            if hasattr(self, "_metrics") and "num_calls" in self._metrics:
                with self._metrics_lock:
                    # Subtract len(audio_paths)-1 since we added 1 in _measure_performance
                    self._metrics["num_calls"] -= (len(audio_paths) - 1)
            
            return results
    
    @abstractmethod
    def from_text_and_image(self, text: str, image_path: str) -> BridgeResult:
        """
        Process text and image together and return structured results.
        
        Args:
            text: Text to process
            image_path: Path to the image file
            
        Returns:
            BridgeResult containing the processed information
        """
        pass
    
    def from_text_and_image_batch(self, texts: List[str], image_paths: List[str]) -> List[BridgeResult]:
        """
        Process a batch of text-image pairs for efficient processing.
        
        This method can be overridden by adapters that support batch processing.
        The default implementation calls from_text_and_image for each pair, but tracks
        total time and tokens for more accurate metrics.
        
        Args:
            texts: List of texts to process
            image_paths: List of paths to image files (must be same length as texts)
            
        Returns:
            List of BridgeResult objects
            
        Raises:
            ValueError: If the lengths of texts and image_paths don't match
        """
        if len(texts) != len(image_paths):
            raise ValueError("The number of texts and image paths must be the same")
        
        with self._measure_performance():
            results = []
            for text, path in zip(texts, image_paths):
                # Don't re-measure performance for individual items
                # since we're already measuring for the whole batch
                try:
                    result = self.from_text_and_image(text, path)
                    results.append(result)
                    
                    # Track tokens for metrics (but don't double-count time)
                    if hasattr(self, "_metrics") and "total_tokens" in self._metrics:
                        with self._metrics_lock:
                            self._metrics["total_tokens"] += len(result.tokens)
                except Exception as e:
                    # Create an error result and continue processing the batch
                    error_result = BridgeResult(
                        tokens=text.split(),
                        labels=[f"Error processing {path}: {str(e)}"]
                    )
                    results.append(error_result)
                    
                    # Track the error in metrics
                    if hasattr(self, "_metrics") and "errors" in self._metrics:
                        with self._metrics_lock:
                            self._metrics["errors"] += 1
            
            # Correct the call count (batch = 1 call, not len(texts) calls)
            if hasattr(self, "_metrics") and "num_calls" in self._metrics:
                with self._metrics_lock:
                    # Subtract len(texts)-1 since we added 1 in _measure_performance
                    self._metrics["num_calls"] -= (len(texts) - 1)
            
            return results
    
    def from_bytes(self, data: Union[bytes, BinaryIO], 
                   mime_type: Optional[str] = None) -> BridgeResult:
        """
        Process binary data directly and return structured results.
        
        This is useful for processing images or audio data without saving to disk.
        
        Args:
            data: Binary data or file-like object with read() method
            mime_type: MIME type of the data (e.g., "image/jpeg", "audio/wav")
            
        Returns:
            BridgeResult containing the processed information
            
        Raises:
            NotImplementedError: If not implemented by the subclass
        """
        raise NotImplementedError(
            "This adapter does not support processing from binary data directly"
        )
    
    def validate_image_path(self, image_path: str) -> str:
        """
        Validate an image file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Validated path
            
        Raises:
            ValueError: If the path is invalid or the file doesn't exist
        """
        import os
        
        if not image_path:
            raise ValueError("Image path cannot be empty")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        if not os.path.isfile(image_path):
            raise ValueError(f"Image path is not a file: {image_path}")
        
        return image_path
    
    def validate_audio_path(self, audio_path: str) -> str:
        """
        Validate an audio file path.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Validated path
            
        Raises:
            ValueError: If the path is invalid or the file doesn't exist
        """
        import os
        
        if not audio_path:
            raise ValueError("Audio path cannot be empty")
        
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        if not os.path.isfile(audio_path):
            raise ValueError(f"Audio path is not a file: {audio_path}")
        
        return audio_path