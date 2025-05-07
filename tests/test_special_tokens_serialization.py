"""
Tests for JSON serialization of special tokens in BridgeResult.
"""

import json
import pytest
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from bridgenlp.result import BridgeResult

class TestSpecialTokensSerialization:
    """Test suite for serialization of special tokens in BridgeResult."""
    
    def test_control_characters(self):
        """Test serialization of control characters."""
        # Create a result with control characters
        control_tokens = ["\x00", "\x01", "\n", "\t", "\r", "\b", "\f", "\x1F"]
        result = BridgeResult(tokens=control_tokens)
        
        # Serialize and deserialize
        json_data = result.to_json()
        json_str = json.dumps(json_data)
        parsed_data = json.loads(json_str)
        
        # All tokens should be preserved, though possibly escaped
        assert len(parsed_data["tokens"]) == len(control_tokens)
        
        # Control characters should be properly escaped but round-trip correctly
        # After JSON serialization, these are escaped but still functionally equivalent
        assert "\n" in parsed_data["tokens"] or "\\n" in parsed_data["tokens"]
        assert "\t" in parsed_data["tokens"] or "\\t" in parsed_data["tokens"]
        assert "\r" in parsed_data["tokens"] or "\\r" in parsed_data["tokens"]
        
    def test_unicode_characters(self):
        """Test serialization of Unicode characters."""
        # Create a result with Unicode characters
        unicode_tokens = ["ñ", "é", "ü", "ø", "ß", "北京", "🔥", "👍", "✓"]
        result = BridgeResult(tokens=unicode_tokens)
        
        # Serialize and deserialize
        json_data = result.to_json()
        json_str = json.dumps(json_data)
        parsed_data = json.loads(json_str)
        
        # Unicode characters should be preserved exactly
        assert parsed_data["tokens"] == unicode_tokens
        
    def test_model_special_tokens(self):
        """Test serialization of special tokens used by models."""
        # Create a result with model special tokens
        model_tokens = [
            "[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]",
            "<s>", "</s>", "<unk>", "<pad>", "<mask>",
            "▁", "##",  # Used by some tokenizers for word pieces
            "<|endoftext|>", "<|startoftext|>"
        ]
        result = BridgeResult(tokens=model_tokens)
        
        # Serialize and deserialize
        json_data = result.to_json()
        json_str = json.dumps(json_data)
        parsed_data = json.loads(json_str)
        
        # Model tokens should be preserved exactly
        assert parsed_data["tokens"] == model_tokens
        
    def test_empty_string(self):
        """Test serialization of empty strings."""
        result = BridgeResult(tokens=["", "non-empty", ""])
        
        # Serialize and deserialize
        json_data = result.to_json()
        json_str = json.dumps(json_data)
        parsed_data = json.loads(json_str)
        
        # Empty strings should be preserved
        assert parsed_data["tokens"] == ["", "non-empty", ""]
        
    def test_backslash_characters(self):
        """Test serialization of backslash characters."""
        # Create a result with backslash characters
        backslash_tokens = ["\\", "\\\\", "\\'", '\\"']
        result = BridgeResult(tokens=backslash_tokens)
        
        # Serialize and deserialize
        json_data = result.to_json()
        json_str = json.dumps(json_data)
        parsed_data = json.loads(json_str)
        
        # The parsed tokens may have extra escapes due to JSON encoding,
        # but when interpreted they should be functionally equivalent
        # Check by re-encoding each as JSON and comparing
        for i, orig_token in enumerate(backslash_tokens):
            # Encode the original token and the parsed token as JSON strings
            orig_json = json.dumps(orig_token)
            parsed_json = json.dumps(parsed_data["tokens"][i])
            # The JSON strings should be equivalent when interpreted
            assert json.loads(orig_json) == json.loads(parsed_json)
            
    def test_json_control_characters(self):
        """Test serialization of JSON control characters."""
        # Create a result with JSON control characters
        json_tokens = ["{", "}", "[", "]", ",", ":", "\""]
        result = BridgeResult(tokens=json_tokens)
        
        # Serialize and deserialize
        json_data = result.to_json()
        json_str = json.dumps(json_data)
        parsed_data = json.loads(json_str)
        
        # JSON control characters should be preserved
        assert parsed_data["tokens"] == json_tokens
        
    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_numpy_types(self):
        """Test serialization of NumPy types."""
        # Skip test if numpy is not available
        if not HAS_NUMPY:
            return
            
        # Create a result with NumPy types
        numpy_tokens = [
            np.int32(42),
            np.float32(3.14159),
            np.array([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            np.array(["a", "b", "c"])
        ]
        
        # Create roles with NumPy values
        roles = [
            {"role": "VALUE", "score": np.float64(0.95), "vector": np.array([0.1, 0.2, 0.3])},
        ]
        
        # Suppress known warnings about conversion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = BridgeResult(tokens=numpy_tokens, roles=roles)
            
            # Serialize and deserialize
            json_data = result.to_json()
            json_str = json.dumps(json_data)
            parsed_data = json.loads(json_str)
            
            # NumPy types should be converted either to strings or native types
            # In our implementation, singular numpy values are converted to strings
            assert parsed_data["tokens"][0] == "42" or parsed_data["tokens"][0] == 42
            assert parsed_data["tokens"][1] == "3.14159" or parsed_data["tokens"][1] == 3.14159 or abs(float(parsed_data["tokens"][1]) - 3.14159) < 1e-5
            
            # Arrays should be converted to lists
            if isinstance(parsed_data["tokens"][2], list):
                assert parsed_data["tokens"][2] == [1, 2, 3]
            else:
                # Or possibly as string representation
                assert "[1, 2, 3]" in parsed_data["tokens"][2]
                
            # 2D arrays
            if isinstance(parsed_data["tokens"][3], list):
                assert parsed_data["tokens"][3] == [[1, 2], [3, 4]]
            else:
                # Or possibly as string representation
                assert "[[1, 2], [3, 4]]" in parsed_data["tokens"][3]
            
            # Check role with NumPy types
            assert parsed_data["roles"][0]["score"] == 0.95
            assert parsed_data["roles"][0]["vector"] == [0.1, 0.2, 0.3]
            
    def test_custom_objects(self):
        """Test serialization of custom objects."""
        # Create a custom object
        class CustomToken:
            def __init__(self, name):
                self.name = name
                
            def __str__(self):
                return f"CustomToken({self.name})"
                
        # Create a mock span object
        class MockSpan:
            def __init__(self, text):
                self.text = text
                
            def __str__(self):
                return self.text
                
        # Create a token function
        def token_func():
            return "function_token"
            
        # Create a result with custom objects
        custom_objects = [
            CustomToken("test"),
            MockSpan("span_text"),
            token_func
        ]
        
        # Suppress known warnings about conversion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = BridgeResult(tokens=custom_objects)
            
            # Serialize and deserialize
            json_data = result.to_json()
            json_str = json.dumps(json_data)
            parsed_data = json.loads(json_str)
            
            # Custom objects should be converted to their string representations
            assert parsed_data["tokens"][0] == "CustomToken(test)"
            assert parsed_data["tokens"][1] == "span_text"  # Uses the .text attribute
            assert "function" in parsed_data["tokens"][2]  # String representation of the function
            
    def test_complex_result(self):
        """Test serialization of a complex result with multiple types."""
        tokens = ["normal", "", "\n", "{", "ü"]
        spans = [(0, 1), (2, 4)]
        clusters = [[(0, 1), (3, 4)]]
        roles = [
            {"role": "TEST", "text": "normal"},
            {"role": "SPECIAL", "text": "\n"}
        ]
        labels = ["LABEL1", "LABEL2", "LABEL3", "LABEL4", "LABEL5"]
        
        result = BridgeResult(
            tokens=tokens,
            spans=spans,
            clusters=clusters,
            roles=roles,
            labels=labels
        )
        
        # Serialize and deserialize
        json_data = result.to_json()
        json_str = json.dumps(json_data)
        parsed_data = json.loads(json_str)
        
        # Complex structure should be preserved
        assert len(parsed_data["tokens"]) == len(tokens)
        assert len(parsed_data["spans"]) == len(spans)
        assert len(parsed_data["clusters"]) == len(clusters)
        assert len(parsed_data["roles"]) == len(roles)
        assert len(parsed_data["labels"]) == len(labels)
        
        # Check that tuples are correctly converted to lists
        assert isinstance(parsed_data["spans"][0], list)
        assert isinstance(parsed_data["clusters"][0][0], list)