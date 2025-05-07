"""
Test the HuggingFaceTranslationBridge adapter.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from bridgenlp.adapters.hf_translation import HuggingFaceTranslationBridge, LanguageDetection
from bridgenlp.config import BridgeConfig
from bridgenlp.result import BridgeResult


class TestHuggingFaceTranslation(unittest.TestCase):
    """Test cases for the HuggingFaceTranslationBridge adapter."""
    
    def setUp(self):
        # Skip tests if required dependencies are not installed
        try:
            import torch
            import transformers
        except ImportError:
            pytest.skip("Required dependencies not installed")
    
    @patch('bridgenlp.adapters.hf_translation.HuggingFaceTranslationBridge._init_model')
    def test_initialization(self, mock_init):
        """Test adapter initialization."""
        # Test default initialization
        adapter = HuggingFaceTranslationBridge()
        self.assertEqual(adapter.model_name, "Helsinki-NLP/opus-mt-en-fr")
        self.assertEqual(adapter.source_lang, "en")
        self.assertEqual(adapter.target_lang, "fr")
        self.assertFalse(adapter.auto_detect_language)
        
        # Test initialization with custom params
        adapter = HuggingFaceTranslationBridge(
            model_name="Helsinki-NLP/opus-mt-de-en",
            source_lang="de",
            target_lang="en",
            auto_detect_language=True
        )
        self.assertEqual(adapter.model_name, "Helsinki-NLP/opus-mt-de-en")
        self.assertEqual(adapter.source_lang, "de")
        self.assertEqual(adapter.target_lang, "en")
        self.assertTrue(adapter.auto_detect_language)
        
        # Verify model is not loaded by default
        mock_init.assert_not_called()
    
    @patch('bridgenlp.adapters.hf_translation.get_or_create_model')
    def test_language_detection_simple(self, mock_get_model):
        """Test basic language detection."""
        # Create adapter with mocked dependencies
        adapter = HuggingFaceTranslationBridge()
        
        # Mock dependencies
        with patch('bridgenlp.adapters.hf_translation.detect_language', return_value="en"):
            # Test basic detection (fallback path)
            detection = adapter.detect_language("Hello world")
            
            # Verify result
            self.assertEqual(detection.code, "en")
            self.assertEqual(detection.name, "English")
            self.assertTrue(detection.supported)  # en should be supported for most models
            
            # Test caching
            # Second call should use cached result
            detection2 = adapter.detect_language("Hello world")
            self.assertEqual(detection, detection2)
    
    @patch('bridgenlp.adapters.hf_translation.fasttext')
    def test_language_detection_fasttext(self, mock_fasttext):
        """Test language detection with FastText."""
        # Create adapter
        adapter = HuggingFaceTranslationBridge()
        
        # Mock FastText predictions
        mock_model = MagicMock()
        mock_model.predict.return_value = ([["__label__fr"]], [[0.95]])
        
        mock_fasttext.load_model.return_value = mock_model
        
        # Mock get_or_create_model to return our mock model
        with patch('bridgenlp.adapters.hf_translation.get_or_create_model', return_value=mock_model):
            # Test detection
            detection = adapter.detect_language("Bonjour le monde")
            
            # Verify detection
            self.assertEqual(detection.code, "fr")
            self.assertEqual(detection.name, "French")
            self.assertEqual(detection.confidence, 0.95)
            self.assertTrue(detection.supported)
    
    @patch('bridgenlp.adapters.hf_translation.langdetect')
    def test_language_detection_langdetect(self, mock_langdetect):
        """Test language detection with langdetect."""
        # Create adapter
        adapter = HuggingFaceTranslationBridge()
        
        # Mock langdetect behavior
        mock_lang = MagicMock()
        mock_lang.lang = "es"
        mock_lang.prob = 0.8
        mock_langdetect.detect_langs.return_value = [mock_lang]
        
        # Mock fasttext to raise import error
        with patch('bridgenlp.adapters.hf_translation.fasttext', side_effect=ImportError):
            # Test detection
            detection = adapter.detect_language("Hola mundo")
            
            # Verify detection
            self.assertEqual(detection.code, "es")
            self.assertEqual(detection.name, "Spanish")
            self.assertEqual(detection.confidence, 0.8)
            self.assertTrue(detection.supported)
    
    def test_get_supported_languages(self):
        """Test extraction of supported languages from model name."""
        # Test Helsinki-NLP model
        adapter = HuggingFaceTranslationBridge(model_name="Helsinki-NLP/opus-mt-en-fr")
        languages = adapter.get_supported_languages()
        self.assertEqual(languages["source_languages"], ["en"])
        self.assertEqual(languages["target_languages"], ["fr"])
        
        # Test multi-language model
        adapter = HuggingFaceTranslationBridge(model_name="Helsinki-NLP/opus-mt-en+de-fr+es")
        languages = adapter.get_supported_languages()
        self.assertEqual(languages["source_languages"], ["en", "de"])
        self.assertEqual(languages["target_languages"], ["fr", "es"])
        
        # Test mBART model
        adapter = HuggingFaceTranslationBridge(model_name="facebook/mbart-large-50-many-to-many-mmt")
        languages = adapter.get_supported_languages()
        self.assertIn("en", languages["source_languages"])
        self.assertIn("fr", languages["source_languages"])
        self.assertIn("ru", languages["source_languages"])
        
        # Test unknown model
        adapter = HuggingFaceTranslationBridge(model_name="custom-model", source_lang="it", target_lang="pt")
        languages = adapter.get_supported_languages()
        self.assertEqual(languages["source_languages"], ["it"])
        self.assertEqual(languages["target_languages"], ["pt"])
    
    def test_language_name_lookup(self):
        """Test language name lookup functionality."""
        adapter = HuggingFaceTranslationBridge()
        
        # Test common language lookups
        self.assertEqual(adapter._get_language_name("en"), "English")
        self.assertEqual(adapter._get_language_name("fr"), "French")
        self.assertEqual(adapter._get_language_name("es"), "Spanish")
        self.assertEqual(adapter._get_language_name("de"), "German")
        
        # Test non-existent language code
        self.assertIsNone(adapter._get_language_name("xx"))
    
    @patch('bridgenlp.adapters.hf_translation.HuggingFaceTranslationBridge._load_model_and_processor')
    @patch('bridgenlp.adapters.hf_translation.HuggingFaceTranslationBridge.detect_language')
    def test_from_text_with_detection(self, mock_detect, mock_load):
        """Test from_text with language detection."""
        # Create adapter
        adapter = HuggingFaceTranslationBridge(auto_detect_language=True)
        
        # Mock dependencies
        mock_detect.return_value = LanguageDetection(
            code="de", 
            name="German", 
            confidence=0.9, 
            supported=True
        )
        
        # Mock translator
        adapter.translator = MagicMock()
        adapter.translator.return_value = [{"translation_text": "Hello world"}]
        
        # Set a dummy tokenization method
        adapter._tokenize_text = lambda x: x.split()
        
        # Test from_text with detection
        result = adapter.from_text("Hallo Welt", detect_lang=True)
        
        # Verify detection was used
        mock_detect.assert_called_once()
        
        # Verify translator was called with correct source language
        self.assertEqual(adapter.translator.call_args[1]["src_lang"], "de")
        
        # Verify result contains detection info
        self.assertEqual(result.roles[0]["source_lang"], "de")
        self.assertEqual(result.roles[0]["detection"]["code"], "de")
        self.assertEqual(result.roles[0]["detection"]["name"], "German")
        self.assertEqual(result.roles[0]["detection"]["confidence"], 0.9)
        self.assertTrue(result.roles[0]["detection"]["supported"])
    
    @patch('bridgenlp.adapters.hf_translation.HuggingFaceTranslationBridge._load_model_and_processor')
    @patch('bridgenlp.adapters.hf_translation.HuggingFaceTranslationBridge.detect_language_batch')
    def test_from_batch_with_detection(self, mock_detect_batch, mock_load):
        """Test from_batch with language detection."""
        # Create adapter
        adapter = HuggingFaceTranslationBridge(auto_detect_language=True)
        
        # Mock dependencies
        mock_detect_batch.return_value = [
            LanguageDetection(code="en", name="English", confidence=0.9, supported=True),
            LanguageDetection(code="fr", name="French", confidence=0.8, supported=True)
        ]
        
        # Mock translator
        adapter.translator = MagicMock()
        adapter.translator.side_effect = lambda **kwargs: [{"translation_text": f"Translated: {kwargs['text']}"}]
        
        # Set a dummy tokenization method
        adapter._tokenize_text = lambda x: x.split()
        
        # Test from_batch with detection
        results = adapter.from_batch(["Hello", "Bonjour"], detect_lang=True)
        
        # Verify batch detection was used
        mock_detect_batch.assert_called_once()
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].roles[0]["source_lang"], "en")
        self.assertEqual(results[0].roles[0]["detection"]["code"], "en")
        self.assertEqual(results[1].roles[0]["source_lang"], "fr")
        self.assertEqual(results[1].roles[0]["detection"]["code"], "fr")
    
    @patch('bridgenlp.adapters.hf_translation.fasttext')
    def test_detect_language_batch(self, mock_fasttext):
        """Test batch language detection."""
        # Create adapter
        adapter = HuggingFaceTranslationBridge()
        
        # Mock FastText predictions
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            [["__label__en"], ["__label__fr"]],  # Labels
            [[0.95], [0.87]]  # Scores
        )
        
        mock_fasttext.load_model.return_value = mock_model
        
        # Mock get_or_create_model to return our mock model
        with patch('bridgenlp.adapters.hf_translation.get_or_create_model', return_value=mock_model):
            # Test batch detection
            detections = adapter.detect_language_batch(["Hello world", "Bonjour le monde"])
            
            # Verify detection results
            self.assertEqual(len(detections), 2)
            self.assertEqual(detections[0].code, "en")
            self.assertEqual(detections[0].confidence, 0.95)
            self.assertEqual(detections[1].code, "fr")
            self.assertEqual(detections[1].confidence, 0.87)


if __name__ == "__main__":
    unittest.main()