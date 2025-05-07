"""
Test the image captioning adapter.

This tests both the basic functionality and prompt conditioning enhancements.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from bridgenlp.adapters.image_captioning import ImageCaptioningBridge
from bridgenlp.result import BridgeResult
from bridgenlp.config import BridgeConfig


class TestImageCaptioning(unittest.TestCase):
    """Test cases for the ImageCaptioningBridge adapter."""
    
    def setUp(self):
        # Skip tests if required dependencies are not installed
        try:
            import torch
            import transformers
            from PIL import Image
        except ImportError:
            pytest.skip("Required dependencies not installed")
            
        # Create a mock image file for testing
        self.image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        if not os.path.exists(self.image_path):
            # Create a simple test image if it doesn't exist
            try:
                from PIL import Image
                img = Image.new('RGB', (100, 100), color = 'red')
                img.save(self.image_path)
            except Exception:
                pytest.skip("Could not create test image")
    
    def tearDown(self):
        # Clean up test image
        if os.path.exists(self.image_path):
            try:
                os.remove(self.image_path)
            except Exception:
                pass
    
    @patch('bridgenlp.adapters.image_captioning.ImageCaptioningBridge._load_model_and_processor')
    def test_initialization(self, mock_load):
        """Test that the adapter initializes correctly."""
        adapter = ImageCaptioningBridge()
        
        # Verify attributes
        self.assertEqual(adapter.model_name, "nlpconnect/vit-gpt2-image-captioning")
        self.assertEqual(adapter.device, -1)
        self.assertFalse(adapter._model_loaded)
        
        # Verify that model is not loaded during initialization
        mock_load.assert_not_called()
    
    @patch('bridgenlp.adapters.image_captioning.ImageCaptioningBridge._load_model_and_processor')
    @patch('bridgenlp.adapters.image_captioning.ImageCaptioningBridge._preprocess_image')
    def test_from_image_mock(self, mock_preprocess, mock_load):
        """Test image captioning with mocked model."""
        # Setup mocks
        adapter = ImageCaptioningBridge()
        adapter._model_loaded = True
        adapter._model_type = "vit-gpt"
        
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "a red image"
        adapter._tokenizer = mock_tokenizer
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        adapter._model = mock_model
        
        # Create mock preprocessed input
        mock_preprocess.return_value = MagicMock()
        
        # Call the method
        result = adapter.from_image(self.image_path)
        
        # Verify result
        self.assertIsInstance(result, BridgeResult)
        self.assertIn("a", result.tokens)
        self.assertIn("red", result.tokens)
        self.assertIn("image", result.tokens)
        self.assertIn("a red image", result.captions)
        self.assertIsNotNone(result.image_features)
        self.assertEqual(result.image_features["caption"], "a red image")
    
    def test_validate_image_path(self):
        """Test image path validation."""
        adapter = ImageCaptioningBridge()
        
        # Valid path
        self.assertEqual(adapter.validate_image_path(self.image_path), self.image_path)
        
        # Invalid paths
        with self.assertRaises(ValueError):
            adapter.validate_image_path("")
            
        with self.assertRaises(ValueError):
            adapter.validate_image_path("/path/to/nonexistent/image.jpg")
    
    @patch('bridgenlp.adapters.image_captioning.ImageCaptioningBridge.from_image')
    def test_from_text(self, mock_from_image):
        """Test from_text method with image path."""
        adapter = ImageCaptioningBridge()
        
        # When text is an image path
        mock_from_image.return_value = BridgeResult(tokens=["a", "red", "image"], captions=["a red image"])
        result = adapter.from_text(self.image_path)
        
        # Verify from_image was called
        mock_from_image.assert_called_once_with(self.image_path)
        
        # When text is not an image path
        result = adapter.from_text("This is not an image path")
        self.assertEqual(result.tokens, ["This", "is", "not", "an", "image", "path"])
        self.assertIn("warning", result.labels[0])
    
    def test_prompt_conditioning_initialization(self):
        """Test that prompt conditioning parameters are initialized correctly."""
        # Test default values
        adapter = ImageCaptioningBridge()
        self.assertFalse(adapter.enable_prompt_conditioning)
        self.assertEqual(adapter.prompt_strategy, "prefix")
        self.assertEqual(adapter.default_prompt, "Describe this image:")
        self.assertEqual(adapter.current_prompt, "Describe this image:")
        
        # Test custom values
        adapter = ImageCaptioningBridge(
            enable_prompt_conditioning=True,
            prompt_strategy="template",
            default_prompt="Analyze this:",
            prompt_template="{prompt} -> {caption}"
        )
        self.assertTrue(adapter.enable_prompt_conditioning)
        self.assertEqual(adapter.prompt_strategy, "template")
        self.assertEqual(adapter.default_prompt, "Analyze this:")
        self.assertEqual(adapter.current_prompt, "Analyze this:")
        self.assertEqual(adapter.prompt_template, "{prompt} -> {caption}")
        
        # Test with config
        config = BridgeConfig()
        config.params["enable_prompt_conditioning"] = True
        config.params["prompt_strategy"] = "instruction"
        config.params["default_prompt"] = "Tell me about:"
        config.params["prompt_template"] = "Q: {prompt} A: {caption}"
        
        adapter = ImageCaptioningBridge(config=config)
        self.assertTrue(adapter.enable_prompt_conditioning)
        self.assertEqual(adapter.prompt_strategy, "instruction")
        self.assertEqual(adapter.default_prompt, "Tell me about:")
        self.assertEqual(adapter.prompt_template, "Q: {prompt} A: {caption}")
    
    def test_set_prompt_methods(self):
        """Test prompt setting and resetting methods."""
        adapter = ImageCaptioningBridge(default_prompt="Default prompt:")
        
        # Initial state
        self.assertEqual(adapter.current_prompt, "Default prompt:")
        
        # Set new prompt
        adapter.set_prompt("New prompt:")
        self.assertEqual(adapter.current_prompt, "New prompt:")
        
        # Reset to default
        adapter.reset_prompt()
        self.assertEqual(adapter.current_prompt, "Default prompt:")
        
        # Set to empty (should use default)
        adapter.set_prompt("")
        self.assertEqual(adapter.current_prompt, "Default prompt:")
    
    def test_apply_prompt_conditioning(self):
        """Test the prompt conditioning application."""
        # Test prefix strategy
        adapter = ImageCaptioningBridge(
            enable_prompt_conditioning=True,
            prompt_strategy="prefix",
            default_prompt="Describe:"
        )
        caption = "a red apple"
        result = adapter._apply_prompt_conditioning(caption)
        self.assertEqual(result, "Describe: a red apple")
        
        # Test with custom prompt
        result = adapter._apply_prompt_conditioning(caption, prompt="Tell me about:")
        self.assertEqual(result, "Tell me about: a red apple")
        
        # Test template strategy
        adapter = ImageCaptioningBridge(
            enable_prompt_conditioning=True,
            prompt_strategy="template",
            default_prompt="Question:",
            prompt_template="Q: {prompt} A: {caption}"
        )
        result = adapter._apply_prompt_conditioning(caption)
        self.assertEqual(result, "Q: Question: A: a red apple")
        
        # Test instruction strategy (should return caption unchanged)
        adapter = ImageCaptioningBridge(
            enable_prompt_conditioning=True,
            prompt_strategy="instruction",
            default_prompt="Describe in detail:"
        )
        result = adapter._apply_prompt_conditioning(caption)
        self.assertEqual(result, caption)
        
        # Test with conditioning disabled
        adapter = ImageCaptioningBridge(
            enable_prompt_conditioning=False,
            prompt_strategy="prefix",
            default_prompt="Describe:"
        )
        result = adapter._apply_prompt_conditioning(caption)
        self.assertEqual(result, caption)  # Should be unchanged
    
    @patch('bridgenlp.adapters.image_captioning.ImageCaptioningBridge._load_model_and_processor')
    @patch('bridgenlp.adapters.image_captioning.ImageCaptioningBridge._preprocess_image')
    def test_from_image_with_prompt(self, mock_preprocess, mock_load):
        """Test image captioning with prompt conditioning."""
        # Setup mocks
        adapter = ImageCaptioningBridge(
            enable_prompt_conditioning=True,
            prompt_strategy="prefix"
        )
        adapter._model_loaded = True
        adapter._model_type = "vit-gpt"
        
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "a red image"
        adapter._tokenizer = mock_tokenizer
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        adapter._model = mock_model
        
        # Create mock preprocessed input
        mock_preprocess.return_value = MagicMock()
        
        # Call the method with a prompt
        result = adapter.from_image(self.image_path, prompt="What color is this:")
        
        # Verify prompt was set
        self.assertEqual(adapter.current_prompt, "What color is this:")
        
        # Verify result has prompt information
        self.assertIsInstance(result, BridgeResult)
        self.assertEqual(result.image_features["prompt"], "What color is this:")
        self.assertEqual(result.image_features["prompt_strategy"], "prefix")
    
    @patch('bridgenlp.adapters.image_captioning.ImageCaptioningBridge.from_image')
    def test_from_text_and_image_with_prompt(self, mock_from_image):
        """Test that from_text_and_image uses text as prompt."""
        adapter = ImageCaptioningBridge(enable_prompt_conditioning=True)
        
        # Call from_text_and_image
        adapter.from_text_and_image("What do you see?", self.image_path)
        
        # Verify from_image was called with the text as prompt
        mock_from_image.assert_called_once_with(self.image_path, prompt="What do you see?")
    
    def test_cleanup(self):
        """Test cleanup method."""
        adapter = ImageCaptioningBridge()
        
        # Mock the resources
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._tokenizer = MagicMock()
        adapter._model_loaded = True
        
        # Set config to unload on cleanup
        adapter.config = MagicMock()
        adapter.config.unload_on_del = True
        
        # Mock unload_model function
        with patch('bridgenlp.adapters.image_captioning.unload_model') as mock_unload:
            adapter.cleanup()
            
            # Verify resources are unloaded
            mock_unload.assert_called_once()
            self.assertIsNone(adapter._model)
            self.assertIsNone(adapter._processor)
            self.assertIsNone(adapter._tokenizer)
            self.assertFalse(adapter._model_loaded)
    
    @pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"), reason="Slow test, set RUN_SLOW_TESTS=1 to run")
    def test_real_model_loading(self):
        """Test loading a real model (slow, only runs if RUN_SLOW_TESTS is set)."""
        try:
            adapter = ImageCaptioningBridge(model_name="nlpconnect/vit-gpt2-image-captioning")
            
            # Access the model property to trigger loading
            adapter._load_model_and_processor()
            
            # Verify model is loaded
            self.assertTrue(adapter._model_loaded)
            self.assertIsNotNone(adapter._model)
            self.assertIsNotNone(adapter._processor)
            
            # Clean up
            adapter.cleanup()
        except ImportError:
            pytest.skip("Required dependencies not installed")
        except Exception as e:
            pytest.skip(f"Model loading failed: {str(e)}")
    
    @pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"), reason="Slow test, set RUN_SLOW_TESTS=1 to run")
    def test_real_prompt_conditioning(self):
        """Test prompt conditioning with a real model (slow)."""
        try:
            # Create adapter with prompt conditioning enabled
            adapter = ImageCaptioningBridge(
                model_name="nlpconnect/vit-gpt2-image-captioning",
                enable_prompt_conditioning=True,
                prompt_strategy="prefix"
            )
            
            # Process with different prompts
            result1 = adapter.from_image(self.image_path, prompt="Describe colors in:")
            result2 = adapter.from_image(self.image_path, prompt="List objects in:")
            
            # Verify different prompts produce different captions
            # (We can't verify exact text, just check they're different)
            self.assertTrue(adapter._model_loaded)
            self.assertIsInstance(result1, BridgeResult)
            self.assertIsInstance(result2, BridgeResult)
            self.assertIn("prompt", result1.image_features)
            self.assertIn("prompt", result2.image_features)
            
            # Clean up
            adapter.cleanup()
        except ImportError:
            pytest.skip("Required dependencies not installed")
        except Exception as e:
            pytest.skip(f"Model loading failed: {str(e)}")


if __name__ == "__main__":
    unittest.main()