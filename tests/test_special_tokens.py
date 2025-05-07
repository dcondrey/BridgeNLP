"""
Test to demonstrate and diagnose the issue with special tokens in JSON serialization.
"""

import json
import pickle
import sys
from bridgenlp.result import BridgeResult

def test_special_tokens_serialization():
    """Test serialization of results containing special tokens."""
    
    # Create some sample data with special tokens that might cause issues
    special_tokens = [
        # Control characters 
        "\n", "\t", "\r", "\b", "\f",
        
        # Unicode characters
        "ñ", "é", "ü", "ø", "ß", "北京", "🔥", "👍", "✓",
        
        # Tokens from transformer models
        "[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]",
        "<s>", "</s>", "<unk>", "<pad>", "<mask>",
        "▁", "##",  # Used by some tokenizers for word pieces
        
        # RoBERTa/GPT style tokens
        "<|endoftext|>", "<|startoftext|>",
        
        # XML-like tokens
        "<entity>", "</entity>",
        
        # Empty string
        "",
        
        # Backslash characters
        "\\", "\\\\", "\\'", '\\"',
        
        # JSON control characters
        "{", "}", "[", "]", ",", ":", "\"",
        
        # Non-printable ASCII
        "\x00", "\x01", "\x1F"
    ]
    
    # Add binary data that might not be representable in JSON
    try:
        # Create tokens with bytes that might not be UTF-8 encodable
        binary_tokens = []
        for i in range(256):
            binary_tokens.append(chr(i))
        
        # Try to add numpy array
        try:
            import numpy as np
            # Use a simple scalar value instead of an array
            numeric_token = np.float32(42.5)
            special_tokens.append(numeric_token)
            print(f"Added numpy float32: {numeric_token}")
        except (ImportError, TypeError) as e:
            print(f"NumPy not available or error: {e}")
        
        # Try to add a custom object
        class CustomToken:
            def __init__(self, name):
                self.name = name
            
            def __str__(self):
                return f"CustomToken({self.name})"
                
        custom_token = CustomToken("test")
        special_tokens.append(custom_token)
        
        # Create a span object to simulate a token from spaCy
        class MockSpan:
            def __init__(self, text):
                self.text = text
                
            def __str__(self):
                return self.text
                
        span_token = MockSpan("span_text")
        special_tokens.append(span_token)
        
        # Try with a callable
        def token_func():
            return "function_token"
            
        special_tokens.append(token_func)
        
        print(f"Added {len(binary_tokens) + 3} special test cases")
        
    except Exception as e:
        print(f"Error setting up advanced test cases: {e}")
    
    # Add binary tokens at the end
    special_tokens.extend(binary_tokens)
    
    # Create a BridgeResult with these special tokens
    result = BridgeResult(
        tokens=special_tokens,
        spans=[(0, 1), (2, 4)],
        clusters=[[(0, 1), (3, 4)]],
        roles=[
            {"role": "SPECIAL", "text": special_tokens[0]},
            {"role": "UNICODE", "text": special_tokens[8]},
            {"role": "MODEL", "text": special_tokens[15]}
        ],
        labels=["SPECIAL" for _ in range(len(special_tokens))]
    )
    
    print("Testing serialization of special tokens:")
    
    # Try to convert to JSON
    try:
        json_data = result.to_json()
        print("✓ to_json() completed without errors")
        
        # Try to serialize to JSON string
        try:
            json_str = json.dumps(json_data)
            print(f"✓ json.dumps() completed without errors - length: {len(json_str)}")
            
            # Try to deserialize back
            try:
                parsed_data = json.loads(json_str)
                print("✓ json.loads() completed without errors")
                
                # Compare token counts to ensure nothing was lost
                print(f"Original token count: {len(special_tokens)}")
                print(f"Serialized token count: {len(parsed_data['tokens'])}")
                
                # Check if all tokens survived round-trip
                for i, (orig, parsed) in enumerate(zip(special_tokens, parsed_data['tokens'])):
                    if orig != parsed:
                        print(f"⚠ Token mismatch at index {i}:")
                        print(f"  Original: {repr(orig)}")
                        print(f"  Parsed:   {repr(parsed)}")
                        
                # Check special characters in roles
                for i, (orig_role, parsed_role) in enumerate(zip(result.roles, parsed_data['roles'])):
                    if orig_role["text"] != parsed_role["text"]:
                        print(f"⚠ Role text mismatch at index {i}:")
                        print(f"  Original: {repr(orig_role['text'])}")
                        print(f"  Parsed:   {repr(parsed_role['text'])}")
                
            except json.JSONDecodeError as e:
                print(f"✗ Error parsing JSON: {e}")
        except TypeError as e:
            print(f"✗ Error during json.dumps(): {e}")
    except Exception as e:
        print(f"✗ Error during to_json(): {e}")

if __name__ == "__main__":
    test_special_tokens_serialization()
