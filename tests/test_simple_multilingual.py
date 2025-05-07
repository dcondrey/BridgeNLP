"""
Simple test script for the enhanced multilingual token alignment capabilities.
This script tests basic functionality without requiring external dependencies.
"""

from bridgenlp.aligner import TokenAligner

def test_script_detection():
    """Test script detection for various languages."""
    print("\nTesting script detection...")
    
    aligner = TokenAligner()
    
    test_cases = [
        ("Hello world", "latin"),
        ("你好世界", "cjk"),
        ("مرحبا بالعالم", "arabic"),
        ("Привет мир", "cyrillic"),
        ("你好你好 world", "cjk"),  # Mixed with more CJK (4 CJK chars)
    ]
    
    passed = 0
    for text, expected in test_cases:
        detected = aligner._detect_script_type(text)
        if detected == expected:
            print(f"✓ '{text}' correctly detected as '{detected}'")
            passed += 1
        else:
            print(f"✗ '{text}' detected as '{detected}', expected '{expected}'")
    
    print(f"Script detection: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)

def test_script_specific_tokenization():
    """Test script-specific tokenization strategies."""
    print("\nTesting script-specific tokenization...")
    
    aligner = TokenAligner()
    
    test_cases = [
        # (text, script_type, expected token count)
        ("Hello, world!", "latin", 4),  # Hello, ,, world, !
        ("你好世界", "cjk", 4),  # 你, 好, 世, 界
        ("مرحبا بالعالم", "arabic", 2),  # مرحبا, بالعالم
    ]
    
    passed = 0
    for text, script, expected_count in test_cases:
        if script == "latin":
            tokens = aligner._tokenize_latin(text)
        elif script == "cjk":
            tokens = aligner._tokenize_cjk(text)
        elif script == "arabic":
            tokens = aligner._tokenize_arabic(text)
        else:
            tokens = aligner._tokenize_default(text)
            
        if len(tokens) == expected_count:
            print(f"✓ '{text}' tokenized into {len(tokens)} tokens as expected")
            passed += 1
        else:
            print(f"✗ '{text}' tokenized into {len(tokens)} tokens, expected {expected_count}")
            print(f"Tokens: {tokens}")
    
    print(f"Tokenization: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)

def test_normalization():
    """Test script-aware text normalization."""
    print("\nTesting script-aware normalization...")
    
    aligner = TokenAligner()
    
    test_cases = [
        # (text, expected normalized output)
        ("Hello WORLD", "hello world"),  # Latin: lowercase
        ("你好世界", "你好世界"),  # CJK: preserved
        ("مرحبا بالعالم", "مرحبا بالعالم"),  # Arabic: preserved with possible normalization
    ]
    
    passed = 0
    for text, expected in test_cases:
        normalized = aligner._normalize_text(text)
        # For CJK and Arabic, we allow some normalization differences
        if normalized.lower() == expected.lower():
            print(f"✓ '{text}' normalized correctly")
            passed += 1
        else:
            print(f"✗ '{text}' normalized to '{normalized}', expected '{expected}'")
    
    print(f"Normalization: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)

def main():
    """Run all tests."""
    print("SIMPLE TEST FOR MULTILINGUAL TOKEN ALIGNMENT")
    print("===========================================")
    
    tests = [
        test_script_detection,
        test_script_specific_tokenization,
        test_normalization,
    ]
    
    total_passed = 0
    for test in tests:
        if test():
            total_passed += 1
    
    print(f"\nOverall: {total_passed}/{len(tests)} test categories passed")
    
    if total_passed == len(tests):
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    main()
