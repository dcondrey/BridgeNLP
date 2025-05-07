"""Test the multilingual capabilities of the translation adapter."""

import pytest
import warnings
from unittest.mock import patch, MagicMock

# Suppress expected warnings about failed alignments in test scenarios
warnings.filterwarnings("ignore", message="Failed to align text segment")

# Import the adapter directly
from bridgenlp.adapters.hf_translation import HuggingFaceTranslationBridge


def test_alignment_generation():
    """Test alignment information generation in the translation adapter."""
    # Create the translator
    translator = HuggingFaceTranslationBridge(lazy_loading=True)
    
    # Set up the metrics to ensure the key exists
    translator._metrics["model_load_time"] = 0.0
    
    # Mock translator to avoid actual model inference
    translator.translator = lambda **kwargs: [{"translation_text": "Ceci est un texte traduit"}]
    
    # Create mock document for alignment
    from bridgenlp.aligner import MockDoc
    mock_doc = MockDoc("This is a test text for alignment testing")
    
    # Test generating alignment information
    source_text = "This is a test text"
    target_text = "Ceci est un texte de test"
    source_script = "latin"
    target_script = "latin"
    
    alignment_info = translator._generate_alignment_info(
        source_text, target_text, source_script, target_script
    )
    
    # Basic validation
    assert isinstance(alignment_info, dict)
    assert "alignments" in alignment_info
    assert "confidence" in alignment_info
    
    print("Alignment info generation test passed!")
    
    # Test with CJK text
    cjk_source = "这是测试文本"
    cjk_target = "これはテストテキストです"
    
    cjk_alignment = translator._generate_alignment_info(
        cjk_source, cjk_target, "cjk", "cjk"
    )
    
    assert isinstance(cjk_alignment, dict)
    assert "alignments" in cjk_alignment
    
    print("CJK alignment generation test passed!")


if __name__ == "__main__":
    # Run the test directly
    test_alignment_generation()
