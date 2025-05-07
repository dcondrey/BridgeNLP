"""
Utility functions for BridgeNLP.

This module provides common utility functions used across the BridgeNLP framework.
"""

import gc
import os
import threading
from typing import Any, Dict, Optional, Type, Union

try:
    import torch
except ImportError:
    torch = None


# Global model registry to avoid loading the same model multiple times
_MODEL_REGISTRY = {}
_MODEL_REGISTRY_LOCK = threading.RLock()


def configure_device(device_config: Union[int, str, None]) -> int:
    """
    Configure the device for model inference.
    
    Args:
        device_config: Device configuration (int GPU index, "cuda", "cpu", or None)
        
    Returns:
        Device index (-1 for CPU, >=0 for GPU)
    """
    # Default to CPU
    device = -1
    
    # Check if torch is available
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return device
    
    # Configure based on the device_config
    if isinstance(device_config, str):
        if device_config.lower() == "cuda":
            device = 0
        elif device_config.isdigit() and int(device_config) >= 0:
            device = int(device_config)
    elif isinstance(device_config, int) and device_config >= 0:
        device = device_config
    
    return device


def get_param_with_fallback(
    direct_value: Any, 
    config: Any, 
    config_attr: str, 
    config_param: Optional[str] = None, 
    default_value: Any = None
) -> Any:
    """
    Get a parameter value with proper fallback logic.
    
    Args:
        direct_value: Directly provided value
        config: Configuration object
        config_attr: Attribute name in the config object
        config_param: Parameter name in config.params (if None, use config_attr)
        default_value: Default value if all else fails
        
    Returns:
        The resolved parameter value
    """
    # Start with the direct value
    if direct_value is not None:
        return direct_value
    
    # Look in configuration if available
    if config:
        # Check direct attribute
        if hasattr(config, config_attr):
            attr_value = getattr(config, config_attr)
            if attr_value is not None:
                return attr_value
                
        # Check in params if requested
        if config_param and hasattr(config, "params"):
            param_value = config.params.get(config_param)
            if param_value is not None:
                return param_value
    
    # Fall back to default
    return default_value


def get_or_create_model(
    model_key: str, 
    model_creator_fn: callable, 
    *args, 
    **kwargs
) -> Any:
    """
    Get a model from the registry or create it if it doesn't exist.
    
    Args:
        model_key: Unique key for the model
        model_creator_fn: Function to create the model if it doesn't exist
        *args: Arguments to pass to the model creator function
        **kwargs: Keyword arguments to pass to the model creator function
        
    Returns:
        The model instance
    """
    with _MODEL_REGISTRY_LOCK:
        if model_key not in _MODEL_REGISTRY:
            _MODEL_REGISTRY[model_key] = model_creator_fn(*args, **kwargs)
        return _MODEL_REGISTRY[model_key]


def unload_model(model_key: str) -> bool:
    """
    Unload a model from the registry.
    
    Args:
        model_key: Key of the model to unload
        
    Returns:
        True if the model was unloaded, False if it wasn't in the registry
    """
    with _MODEL_REGISTRY_LOCK:
        if model_key in _MODEL_REGISTRY:
            _MODEL_REGISTRY.pop(model_key)
            # Force garbage collection to free memory
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False


def free_memory() -> None:
    """
    Force garbage collection and free CUDA memory if available.
    """
    gc.collect()
    if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_text_input(text: str) -> str:
    """
    Validate and clean a text input.
    
    Args:
        text: Input text to validate
        
    Returns:
        Cleaned text
        
    Raises:
        ValueError: If the text is invalid
    """
    if text is None:
        raise ValueError("Text input cannot be None")
    
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Remove null bytes and other problematic characters
    text = text.replace('\0', '')
    
    # Limit length to prevent abuse (1MB is a reasonable limit)
    max_length = 1024 * 1024
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


def detect_language(text: str) -> str:
    """
    Detect the language of a text string.
    
    Args:
        text: Text to detect language for
        
    Returns:
        ISO language code (2-letter) or "en" if detection fails
    """
    try:
        # Import here to avoid hard dependency
        import langdetect
        return langdetect.detect(text)
    except (ImportError, Exception):
        # Fall back to English if language detection fails or is not available
        return "en"


def get_model_memory_usage(model: Any) -> float:
    """
    Estimate the memory usage of a PyTorch model in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Estimated memory usage in MB
    """
    if torch is None:
        return 0.0
        
    try:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        # Convert to MB
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    except Exception:
        return 0.0


def create_model_key(model_name: str, task: str, device: int) -> str:
    """
    Create a unique key for a model in the registry.
    
    Args:
        model_name: Name of the model
        task: Task the model is used for
        device: Device the model is loaded on
        
    Returns:
        Unique model key
    """
    return f"{model_name}_{task}_dev{device}"