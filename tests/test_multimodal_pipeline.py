"""
Test multimodal capabilities in pipelines.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from bridgenlp.pipeline import Pipeline
from bridgenlp.config import BridgeConfig
from bridgenlp.multimodal_base import MultimodalBridgeBase
from bridgenlp.result import BridgeResult


class MockImageAdapter(MultimodalBridgeBase):
    """Mock adapter for testing multimodal capabilities."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.from_image_called = False
        self.from_audio_called = False
        
    def from_image(self, image_path):
        self.from_image_called = True
        return BridgeResult(
            tokens=["image"],
            captions=["a test image"],
            image_features={"path": image_path}
        )
        
    def from_audio(self, audio_path):
        self.from_audio_called = True
        return BridgeResult(
            tokens=["audio"],
            audio_features={"path": audio_path}
        )
        
    def from_text(self, text):
        return BridgeResult(tokens=text.split())
        
    def from_tokens(self, tokens):
        return BridgeResult(tokens=tokens)
        
    def from_spacy(self, doc):
        return doc
        
    def from_text_and_image(self, text, image_path):
        return BridgeResult(
            tokens=text.split(),
            captions=["combined text and image"],
            image_features={"path": image_path}
        )


class TestMultimodalPipeline(unittest.TestCase):
    """Test multimodal capabilities in pipelines."""
    
    def setUp(self):
        # Create mock adapters
        self.image_adapter1 = MockImageAdapter()
        self.image_adapter2 = MockImageAdapter()
        
        # Create a config
        self.config = BridgeConfig()
        self.config.modality = "multimodal"
        
    def test_from_image(self):
        """Test processing an image through a pipeline."""
        # Create a pipeline with both adapters
        pipeline = Pipeline([self.image_adapter1, self.image_adapter2], self.config)
        
        # Process an image
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = pipeline.from_image("test_image.jpg")
        
        # Verify both adapters were called
        self.assertTrue(self.image_adapter1.from_image_called)
        self.assertTrue(self.image_adapter2.from_image_called)
        
        # Verify the result has the expected fields
        self.assertEqual(result.tokens, ["image"])
        self.assertIn("a test image", result.captions)
        self.assertIsNotNone(result.image_features)
        self.assertEqual(result.image_features.get("path"), "test_image.jpg")
        
    def test_from_text_and_image(self):
        """Test processing text and image together through a pipeline."""
        # Create a pipeline with both adapters
        pipeline = Pipeline([self.image_adapter1, self.image_adapter2], self.config)
        
        # Process text and image
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = pipeline.from_text_and_image("test text", "test_image.jpg")
        
        # Verify the result has the expected fields
        self.assertEqual(result.tokens, ["test", "text"])
        self.assertIn("combined text and image", result.captions)
        self.assertIsNotNone(result.image_features)
        self.assertEqual(result.image_features.get("path"), "test_image.jpg")
        
    def test_mixed_pipeline(self):
        """Test a pipeline with both multimodal and non-multimodal adapters."""
        # Create a mock text-only adapter
        text_adapter = MagicMock()
        text_adapter.from_text.return_value = BridgeResult(tokens=["text"], labels=["label"])
        
        # Create a pipeline with mixed adapters
        pipeline = Pipeline([self.image_adapter1, text_adapter], self.config)
        
        # Process an image - should only use multimodal adapters
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isfile", return_value=True):
            result = pipeline.from_image("test_image.jpg")
        
        # Verify only the image adapter was called
        self.assertTrue(self.image_adapter1.from_image_called)
        text_adapter.from_image.assert_not_called()
        
        # Process text - should use all adapters
        result = pipeline.from_text("test text")
        
        # Verify both adapters were used
        text_adapter.from_text.assert_called_with("test text")
        
    def test_invalid_image_path(self):
        """Test error handling for invalid image paths."""
        # Create a pipeline
        pipeline = Pipeline([self.image_adapter1], self.config)
        
        # Test with non-existent file
        with patch("os.path.exists", return_value=False), \
             self.assertRaises(ValueError):
            pipeline.from_image("nonexistent.jpg")
        
        # Test with directory instead of file
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isfile", return_value=False), \
             self.assertRaises(ValueError):
            pipeline.from_image("directory_not_file")
            
    def test_no_compatible_adapters(self):
        """Test error handling when no compatible adapters are found."""
        # Create a mock text-only adapter
        text_adapter = MagicMock()
        text_adapter.from_text.return_value = BridgeResult(tokens=["text"])
        
        # Create a pipeline with only text adapters
        pipeline = Pipeline([text_adapter], self.config)
        
        # Test with image input should raise an error
        with patch("os.path.exists", return_value=True), \
             patch("os.path.isfile", return_value=True), \
             self.assertRaises(ValueError):
            pipeline.from_image("test_image.jpg")


if __name__ == "__main__":
    unittest.main()