"""
Tests for specific bug fixes.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import pytest

from bridgenlp.adapters.multimodal_embeddings import MultimodalEmbeddingsBridge
from bridgenlp.result import BridgeResult
from bridgenlp.multimodal_base import MultimodalBridgeBase


class TestMultimodalEmbeddingsBugfix(unittest.TestCase):
    """Tests for bug fixes in MultimodalEmbeddingsBridge."""
    
    def test_combined_embedding_division_fix(self):
        """
        Test the fix for dividing a list by 2 in combined embeddings.
        
        This tests the fix for the bug where (text_embeds + image_embeds).tolist() / 2
        was incorrect (trying to divide a list by 2).
        """
        # Skip if numpy is not available
        try:
            import numpy as np
        except ImportError:
            pytest.skip("NumPy not installed")
            
        # Create a mock adapter with minimal implementation
        adapter = MultimodalEmbeddingsBridge()
        
        # Create mock numpy arrays
        text_embeds = np.array([2.0, 4.0, 6.0])
        image_embeds = np.array([4.0, 6.0, 8.0])
        
        # Create mock outputs object with these arrays
        outputs = MagicMock()
        outputs.text_embeds = text_embeds.reshape(1, -1)
        outputs.image_embeds = image_embeds.reshape(1, -1)
        
        # Mock the model call to return our mock outputs
        with patch.object(adapter, '_model') as mock_model, \
             patch.object(adapter, '_model_loaded', True), \
             patch.object(adapter, '_process_text_and_image', return_value={}), \
             patch.object(adapter, 'validate_image_path', return_value="test.jpg"), \
             patch('bridgenlp.utils.validate_text_input', return_value="test"):
            
            mock_model.return_value = outputs
            
            # Mock torch.no_grad context to just yield
            with patch('torch.no_grad', MagicMock()):
                # Call method that previously had the bug
                with patch('torch.matmul', return_value=MagicMock(item=lambda: 0.95)):
                    result = adapter.from_text_and_image("test", "test.jpg")
            
        # Verify the division is done correctly
        # Expected: ([2+4, 4+6, 6+8] / 2) = [3, 5, 7]
        expected = [3.0, 5.0, 7.0]
        self.assertIsNotNone(result.multimodal_embeddings)
        
        # Assert all elements are close to the expected values
        for actual, expected_val in zip(result.multimodal_embeddings, expected):
            self.assertAlmostEqual(actual, expected_val, places=5)


class TestAbstractMethodImplementation(unittest.TestCase):
    """Tests for the abstract method implementations."""
    
    def test_from_audio_implementation(self):
        """
        Test that from_audio is properly implemented in all adapters.
        
        This tests the fix for implementing from_audio which is an abstract
        method in MultimodalBridgeBase.
        """
        # Import all multimodal adapters
        try:
            from bridgenlp.adapters.image_captioning import ImageCaptioningBridge
            from bridgenlp.adapters.object_detection import ObjectDetectionBridge
            from bridgenlp.adapters.multimodal_embeddings import MultimodalEmbeddingsBridge
        except ImportError:
            pytest.skip("Multimodal adapters not installed")
        
        # Create instances of each adapter
        adapters = [
            ImageCaptioningBridge(),
            ObjectDetectionBridge(),
            MultimodalEmbeddingsBridge()
        ]
        
        # Check if from_audio is implemented in each
        for adapter in adapters:
            # Verify the method is implemented
            self.assertTrue(hasattr(adapter, 'from_audio'))
            self.assertTrue(callable(adapter.from_audio))
            
            # Patch the validation to avoid file not found errors
            with patch.object(adapter, 'validate_audio_path', return_value="test.wav"):
                # Call the method to ensure it doesn't raise AbstractMethodError
                result = adapter.from_audio("test.wav")
                
                # Verify it returns a BridgeResult
                self.assertIsInstance(result, BridgeResult)
                
                # Verify it contains a warning
                self.assertIn("warning", result.labels[0])


if __name__ == "__main__":
    unittest.main()