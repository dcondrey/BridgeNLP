"""
Tests for the configuration system.
"""

import json
import os
import sys
import tempfile
import pytest

# Add the parent directory to the path so we can import bridgenlp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bridgenlp.config import BridgeConfig


class TestBridgeConfig:
    """Test suite for BridgeConfig class."""
    
    def test_init(self):
        """Test basic initialization."""
        config = BridgeConfig(model_type="test")
        assert config.model_type == "test"
        assert config.model_name is None
        assert config.device == -1
        assert config.batch_size == 1
        assert config.collect_metrics is False
    
    def test_init_with_values(self):
        """Test initialization with custom values."""
        config = BridgeConfig(
            model_type="sentiment",
            model_name="test-model",
            device="cuda",
            batch_size=8,
            collect_metrics=True
        )
        assert config.model_type == "sentiment"
        assert config.model_name == "test-model"
        assert config.device == "cuda"
        assert config.batch_size == 8
        assert config.collect_metrics is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = BridgeConfig(
            model_type="test",
            model_name="model",
            device=0
        )
        config_dict = config.to_dict()
        assert config_dict["model_type"] == "test"
        assert config_dict["model_name"] == "model"
        assert config_dict["device"] == 0
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "model_type": "test",
            "model_name": "model",
            "device": "cuda",
            "custom_param": "value"
        }
        config = BridgeConfig.from_dict(config_dict)
        assert config.model_type == "test"
        assert config.model_name == "model"
        assert config.device == "cuda"
        assert "custom_param" not in vars(config)
        assert config.params["custom_param"] == "value"
    
    def test_to_json_from_json(self):
        """Test serialization to and from JSON."""
        config = BridgeConfig(
            model_type="test",
            model_name="model",
            device=0,
            batch_size=4
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name
        
        try:
            # Write to file
            config.to_json(tmp_path)
            
            # Read from file
            loaded_config = BridgeConfig.from_json(tmp_path)
            
            # Verify
            assert loaded_config.model_type == config.model_type
            assert loaded_config.model_name == config.model_name
            assert loaded_config.device == config.device
            assert loaded_config.batch_size == config.batch_size
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_invalid_device(self):
        """Test validation of device parameter."""
        with pytest.raises(ValueError):
            BridgeConfig.from_dict({
                "model_type": "test",
                "device": "invalid_device"
            })
    
    def test_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            BridgeConfig.from_json("/path/to/nonexistent/file.json")

    def test_env_overrides_from_dict(self, monkeypatch):
        """Environment variables override dictionary values."""
        monkeypatch.setenv("BRIDGENLP_DEVICE", "cuda")
        monkeypatch.setenv("BRIDGENLP_BATCH_SIZE", "5")
        config = BridgeConfig.from_dict({
            "model_type": "test",
            "device": "cpu",
            "batch_size": 1,
        })
        assert config.device == "cuda"
        assert config.batch_size == 5

    def test_env_overrides_from_json(self, monkeypatch, tmp_path):
        """Environment variables override JSON config values."""
        cfg = {
            "model_type": "test",
            "device": -1,
            "batch_size": 2,
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))
        monkeypatch.setenv("BRIDGENLP_DEVICE", "0")
        monkeypatch.setenv("BRIDGENLP_BATCH_SIZE", "3")
        config = BridgeConfig.from_json(str(path))
        assert config.device == 0
        assert config.batch_size == 3
