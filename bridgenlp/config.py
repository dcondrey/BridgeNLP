"""
Configuration system for BridgeNLP adapters.

This module provides a standardized way to configure and initialize
bridge adapters with consistent options.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class BridgeConfig:
    """
    Configuration container for bridge adapters and pipelines.
    
    This class provides a standardized way to configure bridge adapters
    and pipelines with consistent options and serialization capabilities.
    """
    
    # Model configuration
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    device: Union[int, str] = -1  # -1 for CPU, >=0 for specific GPU, or "cuda"/"cpu"
    
    # Modality configuration
    modality: str = "text"  # Options: "text", "image", "audio", "multimodal"
    
    # Performance options
    batch_size: int = 1
    max_length: Optional[int] = None
    use_threading: bool = False
    num_threads: int = 4
    
    # Resource management
    unload_on_del: bool = True
    cache_size: int = 100  # For LRU caches
    
    # Result caching
    cache_results: bool = False  # Enable result caching for pipelines
    
    # Metrics collection
    collect_metrics: bool = False
    
    # Pipeline options
    pipeline_parallel: bool = False  # Run pipeline stages in parallel when possible
    pipeline_timeout: Optional[float] = None  # Timeout for pipeline operations in seconds
    
    # Image processing options
    image_size: Optional[Dict[str, int]] = None  # e.g., {"height": 224, "width": 224}
    image_processor: Optional[str] = None
    
    # Audio processing options
    audio_sample_rate: int = 16000
    max_audio_length: Optional[float] = None
    audio_processor: Optional[str] = None
    
    # Model-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BridgeConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            BridgeConfig instance
            
        Raises:
            ValueError: If the configuration contains invalid values
        """
        # Extract known fields
        known_fields = {k for k in cls.__dataclass_fields__ if k != "params"}
        base_config = {k: v for k, v in config_dict.items() if k in known_fields}
        
        # Put remaining fields in params
        params = {k: v for k, v in config_dict.items() if k not in known_fields}
        
        # Create instance
        config = cls(**base_config)
        config.params = params
        
        # Validate device
        if isinstance(config.device, str) and config.device not in ["cpu", "cuda", "-1"]:
            try:
                # Try to convert string to int for GPU index
                config.device = int(config.device)
            except ValueError:
                raise ValueError(f"Invalid device value: {config.device}. "
                                 f"Must be an integer, 'cpu', 'cuda', or '-1'")
        
        return config
    
    @classmethod
    def from_json(cls, json_path: str) -> "BridgeConfig":
        """
        Load configuration from a JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            BridgeConfig instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the configuration is invalid
        """
        try:
            with open(json_path, "r") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {json_path}", e.doc, e.pos)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        config_dict = asdict(self)
        
        # Merge params into the top level
        params = config_dict.pop("params", {})
        config_dict.update(params)
        
        return config_dict
    
    def to_json(self, json_path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            json_path: Path to save the JSON configuration
        """
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist and path contains directories
        dir_path = os.path.dirname(os.path.abspath(json_path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=2)
