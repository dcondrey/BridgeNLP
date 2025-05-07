"""
Multilingual Alignment Demo

This script demonstrates the enhanced multilingual capabilities of BridgeNLP's token alignment,
focusing on script-specific handling for different writing systems.
"""

from bridgenlp.aligner import TokenAligner
from bridgenlp.adapters.hf_translation import HuggingFaceTranslationBridge
from bridgenlp.result import BridgeResult

def demonstrate_script_detection():
    """Demonstrate automatic script detection for various languages."""
    print("\n===== SCRIPT DETECTION =====")
    
    aligner = TokenAligner()
    
    test_texts = {
        "English": "This is a test document.",
        "French": "Ceci est un document de test.",
        "Chinese": "这是一个测试文档。",
        "Japanese": "これはテスト文書です。",
        "Korean": "이것은 테스트 문서입니다.",
        "Arabic": "هذا مستند اختبار.",
        "Russian": "Это тестовый документ.",
        "Hindi": "यह एक परीक्षण दस्तावेज़ है।",
        "Mixed": "This is 测试 document with مختلط scripts."
    }
    
    for language, text in test_texts.items():
        script_type = aligner._detect_script_type(text)
        print(f"{language}: '{text}' → Script type: '{script_type}'")

def demonstrate_script_specific_tokenization():
    """Demonstrate script-specific tokenization strategies."""
    print("\n===== SCRIPT-SPECIFIC TOKENIZATION =====")
    
    aligner = TokenAligner()
    
    test_texts = {
        "Latin (English)": "Hello, world! This is a test.",
        "CJK (Chinese)": "你好，世界！这是一个测试。",
        "Arabic": "مرحبا بالعالم! هذا اختبار.",
        "Cyrillic (Russian)": "Привет, мир! Это тест."
    }
    
    for language, text in test_texts.items():
        script_type = aligner._detect_script_type(text)
        print(f"\n{language} (Script: {script_type}):")
        print(f"Original: '{text}'")
        
        # Use script-specific tokenization
        if script_type == "latin" or script_type == "cyrillic":
            tokens = aligner._tokenize_latin(text)
        elif script_type == "cjk":
            tokens = aligner._tokenize_cjk(text)
        elif script_type == "arabic":
            tokens = aligner._tokenize_arabic(text)
        else:
            tokens = aligner._tokenize_default(text)
            
        print(f"Tokens ({len(tokens)}): {tokens}")

def demonstrate_multilingual_alignment():
    """Demonstrate alignment between text segments in different scripts."""
    print("\n===== MULTILINGUAL ALIGNMENT =====")
    
    aligner = TokenAligner()
    
    # Example documents in different scripts
    docs = {
        "latin": aligner._get_spacy_doc(
            "This is an example document for testing multilingual alignment capabilities."
        ),
        "cjk": aligner._get_spacy_doc(
            "这是一个测试多语言对齐功能的示例文档。"
        ),
        "arabic": aligner._get_spacy_doc(
            "هذه وثيقة مثال لاختبار قدرات المحاذاة متعددة اللغات."
        ),
        "cyrillic": aligner._get_spacy_doc(
            "Это пример документа для тестирования возможностей многоязычного выравнивания."
        )
    }
    
    # Test segments to align in each script
    segments = {
        "latin": "example document",
        "cjk": "示例文档",
        "arabic": "وثيقة مثال",
        "cyrillic": "пример документа"
    }
    
    # Demonstrate alignment for each script
    for script_type, doc in docs.items():
        segment = segments[script_type]
        print(f"\nAligning in {script_type.upper()} script:")
        print(f"Document: '{doc.text}'")
        print(f"Segment:  '{segment}'")
        
        # Apply fuzzy alignment with script awareness
        span = aligner.fuzzy_align(doc, segment, script_type=script_type)
        
        if span:
            print(f"Match found: '{span.text}'")
            # Calculate similarity score
            if script_type == "latin" or script_type == "cyrillic":
                tokens = aligner._tokenize_latin(segment)
                score_method = aligner._score_latin
            elif script_type == "cjk":
                tokens = aligner._tokenize_cjk(segment)
                score_method = aligner._score_latin  # Fallback to avoid errors
            elif script_type == "arabic":
                tokens = aligner._tokenize_arabic(segment)
                score_method = aligner._score_latin  # Fallback to avoid errors
            else:
                tokens = aligner._tokenize_default(segment)
                score_method = aligner._score_default
                
            similarity = aligner._calculate_similarity_score(span, tokens, script_type)
            print(f"Similarity score: {similarity:.2f}")
        else:
            print("No match found")

def demonstrate_translation_alignment():
    """Demonstrate alignment between source and translated text."""
    print("\n===== TRANSLATION ALIGNMENT =====")
    
    try:
        # Create a translation bridge
        translator = HuggingFaceTranslationBridge(
            model_name="Helsinki-NLP/opus-mt-en-fr",
            source_lang="en",
            target_lang="fr"
        )
        
        # Translate with alignment information
        text = "This is a multilingual test document demonstrating alignment capabilities."
        print(f"Source text (English): '{text}'")
        
        result = translator.from_text(text)
        
        # Extract translation and alignment info
        translation = result.roles[0]["text"]
        print(f"Translation (French): '{translation}'")
        
        alignment_info = result.roles[0]["alignment"]
        
        # Display alignment between source and target
        print("\nAlignment between source and target:")
        for align in alignment_info["alignments"]:
            source_indices = align["source_indices"]
            target_indices = align["target_indices"]
            score = align["score"]
            
            if source_indices and target_indices:
                source_tokens = [alignment_info["source_tokens"][i] for i in source_indices]
                target_tokens = [alignment_info["target_tokens"][i] for i in target_indices]
                
                print(f"Source: '{' '.join(source_tokens)}' → Target: '{' '.join(target_tokens)}' (score: {score:.2f})")
        
        print(f"\nOverall alignment confidence: {alignment_info['confidence']:.2f}")
        
    except ImportError:
        print("Translation demonstration requires Hugging Face transformers and torch.")
        print("Install with: pip install transformers torch")

def main():
    """Run all demonstrations."""
    print("BRIDGENLP MULTILINGUAL ALIGNMENT DEMONSTRATION")
    print("=============================================")
    
    # Demonstrate script detection
    demonstrate_script_detection()
    
    # Demonstrate script-specific tokenization
    demonstrate_script_specific_tokenization()
    
    # Demonstrate multilingual alignment
    demonstrate_multilingual_alignment()
    
    # Demonstrate translation alignment
    demonstrate_translation_alignment()

if __name__ == "__main__":
    main()