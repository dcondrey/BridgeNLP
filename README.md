# BridgeNLP

A minimal, robust, universal AI model-to-pipeline integration framework.

## What is BridgeNLP?

BridgeNLP serves as a **universal adapter layer** between advanced AI models (e.g., AllenNLP, Hugging Face) and structured pipelines (e.g., spaCy). Its core goal is to allow developers to integrate models for various tasks like NLP (e.g., coreference resolution, semantic role labeling) and multimodal processing (e.g., image captioning, object detection) into applications in a clean, aligned, and memory-safe manner.

Key features:
- **Minimal dependencies**: Only requires spaCy and NumPy by default
- **Modular and extensible**: Add only the model adapters you need
- **Multimodal support**: Process text, images, and combined text-image inputs
- **Token alignment**: Seamlessly map between different tokenization schemes with multilingual support
- **Memory-efficient**: Careful resource management and minimal copying
- **Well-documented and tested**: Production-ready code with comprehensive tests

## Installation

### Basic Installation

```bash
pip install bridgenlp
```

### With Optional Extras

For AllenNLP support (coreference resolution):
```bash
pip install bridgenlp[allennlp]
```

For Hugging Face support (semantic role labeling, text generation):
```bash
pip install bridgenlp[huggingface]
```

For multimodal support (image captioning, object detection):
```bash
pip install bridgenlp[multimodal]
```

For all features:
```bash
pip install bridgenlp[all]
```

## Usage Examples

### Using with spaCy Pipelines

```python
import spacy
from bridgenlp.adapters.allen_coref import AllenNLPCorefBridge
from bridgenlp.pipes.spacy_pipe import SpacyBridgePipe

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Create a bridge adapter
coref_bridge = AllenNLPCorefBridge()

# Add as a pipeline component
nlp.add_pipe("bridgenlp", config={"bridge": coref_bridge})

# Process text
doc = nlp("Julie hugged David because she missed him.")

# Access results
for cluster in doc._.nlp_bridge_clusters:
    print("Coreference cluster:")
    for start, end in cluster:
        print(f"  - {doc[start:end].text}")
```

### Using the CLI

```bash
# Process a single text
bridgenlp predict --model coref --text "Julie hugged David because she missed him."

# Process a file
bridgenlp predict --model srl --file input.txt --output results.json

# Process stdin to stdout
cat input.txt | bridgenlp predict --model coref > output.json

# Use named entity recognition
bridgenlp predict --model ner --text "Apple is looking at buying U.K. startup for $1 billion."

# Pretty-print the output
bridgenlp predict --model ner --text "Apple is looking at buying U.K. startup for $1 billion." --pretty
```

### Programmatic Adapter Use

```python
from bridgenlp.adapters.hf_srl import HuggingFaceSRLBridge

# Create a bridge adapter
srl_bridge = HuggingFaceSRLBridge()

# Process text directly
result = srl_bridge.from_text("Julie hugged David because she missed him.")

# Access results
for role in result.roles:
    print(f"{role['role']}: {role['text']}")

# Convert to JSON
json_data = result.to_json()
```

## Available Adapters

BridgeNLP provides several adapters for different NLP and multimodal tasks:

### Text Generation

#### Text Summarization

```python
from bridgenlp.adapters.hf_summarization import HuggingFaceSummarizationBridge

# Create a summarization bridge
summarizer = HuggingFaceSummarizationBridge(
    model_name="facebook/bart-large-cnn",  # Default model
    max_length=130,  # Maximum length of generated summary
    min_length=30,   # Minimum length of generated summary
    device=0  # Use GPU (0) or CPU (-1)
)

# Process text
result = summarizer.from_text("Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.")

# Access summarization result
summary = result.roles[0]["text"]
print(f"Summary: {summary}")
```

#### Text Paraphrasing

```python
from bridgenlp.adapters.hf_paraphrase import HuggingFaceParaphraseBridge

# Create a paraphrasing bridge
paraphraser = HuggingFaceParaphraseBridge(
    model_name="tuner007/pegasus_paraphrase",  # Default model
    num_return_sequences=3,  # Generate multiple variations
    temperature=0.7,  # Control randomness (higher = more creative)
    device=0  # Use GPU (0) or CPU (-1)
)

# Process text
result = paraphraser.from_text("Artificial intelligence is transforming the way we live and work.")

# Access paraphrases
for role in result.roles:
    print(f"Paraphrase {role['variant']}: {role['text']}")
```

#### Translation

```python
from bridgenlp.adapters.hf_translation import HuggingFaceTranslationBridge

# Create a translation bridge (English to French)
translator = HuggingFaceTranslationBridge(
    model_name="Helsinki-NLP/opus-mt-en-fr",  # Model determines language pair
    device=0  # Use GPU (0) or CPU (-1)
)

# Process text
result = translator.from_text("Artificial intelligence is transforming the way we live and work.")

# Access translation
translation = result.roles[0]["text"]
source_lang = result.roles[0]["source_lang"]
target_lang = result.roles[0]["target_lang"]
print(f"Translation ({source_lang} → {target_lang}): {translation}")
```

### Coreference Resolution

```python
from bridgenlp.adapters.allen_coref import AllenNLPCorefBridge

# Create a coreference resolution bridge
coref_bridge = AllenNLPCorefBridge(
    model_name="coref-spanbert",  # Default model
    device=0  # Use GPU (0) or CPU (-1)
)

# Process text
result = coref_bridge.from_text("Julie hugged David because she missed him.")

# Access coreference clusters
for cluster in result.clusters:
    print("Coreference cluster:")
    for start, end in cluster:
        print(f"  - {result.tokens[start:end]}")
```

### Semantic Role Labeling

```python
from bridgenlp.adapters.hf_srl import HuggingFaceSRLBridge

# Create a semantic role labeling bridge
srl_bridge = HuggingFaceSRLBridge(
    model_name="Davlan/bert-base-multilingual-cased-srl-nli",  # Default model
    device=0  # Use GPU (0) or CPU (-1)
)

# Process text
result = srl_bridge.from_text("Julie hugged David because she missed him.")

# Access semantic roles
for role in result.roles:
    print(f"{role['role']}: {role['text']} (score: {role['score']:.2f})")
```

### Named Entity Recognition

```python
from bridgenlp.adapters.spacy_ner import SpacyNERBridge

# Create a named entity recognition bridge
ner_bridge = SpacyNERBridge(
    model_name="en_core_web_sm"  # Default spaCy model
)

# Process text
result = ner_bridge.from_text("Apple is looking at buying U.K. startup for $1 billion.")

# Access named entities
for i, label in enumerate(result.labels):
    if label != "O":  # "O" means outside any entity
        print(f"{label}: {result.tokens[i]}")
```

### Sentiment Analysis

```python
from bridgenlp.adapters.hf_sentiment import HuggingFaceSentimentBridge

# Create a sentiment analysis bridge
sentiment_bridge = HuggingFaceSentimentBridge(
    model_name="distilbert-base-uncased-finetuned-sst-2-english",  # Default model
    device=0  # Use GPU (0) or CPU (-1)
)

# Process text
result = sentiment_bridge.from_text("I love this product! It's amazing.")

# Access sentiment results
for role in result.roles:
    print(f"Sentiment: {role['label']} (confidence: {role['score']:.2f})")
```

### Text Classification

```python
from bridgenlp.adapters.hf_classification import HuggingFaceClassificationBridge

# Create a text classification bridge with custom labels
classification_bridge = HuggingFaceClassificationBridge(
    model_name="facebook/bart-large-mnli",  # Default model
    device=0,  # Use GPU (0) or CPU (-1)
    labels=["politics", "sports", "technology"]  # Custom labels
)

# Process text
result = classification_bridge.from_text("The new iPhone was announced yesterday.")

# Access classification results
for role in result.roles:
    print(f"Class: {role['label']} (confidence: {role['score']:.2f})")
```

### Question Answering

```python
from bridgenlp.adapters.hf_qa import HuggingFaceQABridge

# Create a question answering bridge
qa_bridge = HuggingFaceQABridge(
    model_name="deepset/roberta-base-squad2",  # Default model
    device=0  # Use GPU (0) or CPU (-1)
)

# Set the question
qa_bridge.set_question("Who built the Eiffel Tower?")

# Process context text
result = qa_bridge.from_text("The Eiffel Tower was built by Gustave Eiffel's company.")

# Access answer
for role in result.roles:
    print(f"Answer: {role['text']} (confidence: {role['score']:.2f})")
```

### NLTK Integration

```python
from bridgenlp.adapters.nltk_adapter import NLTKBridge

# Create an NLTK bridge
nltk_bridge = NLTKBridge(use_pos=True, use_ner=True)

# Process text
result = nltk_bridge.from_text("Apple Inc. is based in Cupertino.")

# Access POS tags and named entities
for token, tag in zip(result.tokens, result.labels):
    print(f"{token}: {tag}")

for start, end in result.spans:
    entity = " ".join(result.tokens[start:end])
    entity_type = result.labels[start]
    print(f"{entity_type}: {entity}")
```

## Multimodal Adapters

BridgeNLP now supports multimodal processing with these adapters:

### Image Captioning

```python
from bridgenlp.adapters.image_captioning import ImageCaptioningBridge

# Create an image captioning bridge
captioning = ImageCaptioningBridge(
    model_name="nlpconnect/vit-gpt2-image-captioning",  # Default model
    device=0,  # Use GPU (0) or CPU (-1)
    num_captions=3  # Generate multiple captions
)

# Process an image
result = captioning.from_image("path/to/image.jpg")

# Access captions
for i, caption in enumerate(result.captions):
    print(f"Caption {i+1}: {caption}")

# Access image features
image_path = result.image_features["image_path"]
print(f"Image path: {image_path}")
```

### Object Detection

```python
from bridgenlp.adapters.object_detection import ObjectDetectionBridge

# Create an object detection bridge
detection = ObjectDetectionBridge(
    model_name="facebook/detr-resnet-50",  # Default model
    device=0,  # Use GPU (0) or CPU (-1)
    threshold=0.7  # Confidence threshold for detections
)

# Process an image
result = detection.from_image("path/to/image.jpg")

# Access detected objects
print(f"Found {len(result.detected_objects)} objects:")
for obj in result.detected_objects:
    label = obj["label"]
    score = obj["score"]
    box = obj["box"]  # [x1, y1, x2, y2]
    print(f"- {label} (confidence: {score:.2f}, bbox: {box})")

# Access auto-generated caption
if result.captions:
    print(f"Auto-caption: {result.captions[0]}")
```

### Multimodal Embeddings

```python
from bridgenlp.adapters.multimodal_embeddings import MultimodalEmbeddingsBridge

# Create a multimodal embeddings bridge
embeddings = MultimodalEmbeddingsBridge(
    model_name="openai/clip-vit-base-patch32",  # Default model
    device=0  # Use GPU (0) or CPU (-1)
)

# Process text and image together
result = embeddings.from_text_and_image(
    "a dog playing in the park",
    "path/to/image.jpg"
)

# Access similarity score between text and image
similarity = result.roles[0]["similarity_score"]
print(f"Text-image similarity: {similarity:.4f}")

# Access embeddings
text_embedding = result.roles[0]["text_embedding"]
image_embedding = result.image_features["embedding"]
multimodal_embedding = result.multimodal_embeddings

# Calculate similarity between different images and text
similarity = embeddings.calculate_similarity(
    "a cat sleeping on a couch",
    "path/to/other_image.jpg"
)
print(f"Text-image similarity: {similarity:.4f}")
```

## Advanced Usage

### Pipeline Composition

The `Pipeline` class allows combining multiple adapters into a single processing pipeline for both text and multimodal inputs:

```python
import spacy
from bridgenlp.pipeline import Pipeline
from bridgenlp.adapters.spacy_ner import SpacyNERBridge
from bridgenlp.adapters.hf_sentiment import HuggingFaceSentimentBridge
from bridgenlp.config import BridgeConfig

# Create a spaCy pipeline
nlp = spacy.load("en_core_web_sm")

# Create individual adapters
ner_bridge = SpacyNERBridge(model_name="en_core_web_sm")
config = BridgeConfig(cache_results=True)  # Enable result caching
sentiment_bridge = HuggingFaceSentimentBridge(config=config)

# Create a pipeline with both adapters
pipeline = Pipeline([ner_bridge, sentiment_bridge], config)

# Process a text with the full pipeline
text = "Apple is looking at buying U.K. startup for $1 billion. It would be a great acquisition."
doc = nlp(text)
doc = pipeline.from_spacy(doc)

# Access named entities
for span_start, span_end in doc._.nlp_bridge_spans:
    label = doc._.nlp_bridge_labels[span_start]
    if label != "O":
        print(f"{label}: {doc[span_start:span_end].text}")

# Access sentiment
for role in doc._.nlp_bridge_roles:
    if "label" in role and "score" in role:
        print(f"Sentiment: {role['label']} (confidence: {role['score']:.2f})")
```

### Multimodal Pipeline

The pipeline can also combine multimodal adapters:

```python
from bridgenlp.pipeline import Pipeline
from bridgenlp.adapters.image_captioning import ImageCaptioningBridge
from bridgenlp.adapters.object_detection import ObjectDetectionBridge
from bridgenlp.config import BridgeConfig

# Create a multimodal config
config = BridgeConfig(modality="multimodal", device=0)

# Create individual adapters
captioning = ImageCaptioningBridge(config=config)
detection = ObjectDetectionBridge(config=config)

# Create a pipeline with both adapters
pipeline = Pipeline([captioning, detection], config)

# Process an image with the full pipeline
result = pipeline.from_image("path/to/image.jpg")

# Access captions
for caption in result.captions:
    print(f"Caption: {caption}")

# Access detected objects
for obj in result.detected_objects:
    print(f"Detected: {obj['label']} ({obj['score']:.2f})")
```

### Token Alignment

The `TokenAligner` class provides utilities for aligning tokens between different tokenization schemes:

```python
import spacy
from bridgenlp.aligner import TokenAligner

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a test document.")

aligner = TokenAligner()

# Align by character span
span = aligner.align_char_span(doc, 10, 14)  # "test"
print(span.text)  # "test"

# Align by token span from a different tokenization
model_tokens = ["This", "is", "a", "test", "document", "."]
span = aligner.align_token_span(doc, 3, 5, model_tokens)  # "test document"
print(span.text)  # "test document"

# Fuzzy alignment for approximate matches
span = aligner.fuzzy_align(doc, "TEST document")
print(span.text)  # "test document"
```

### Memory Management

BridgeNLP is designed to minimize memory usage. Here are some best practices:

```python
# Create a bridge adapter
from bridgenlp.adapters.allen_coref import AllenNLPCorefBridge
coref_bridge = AllenNLPCorefBridge()

# Process documents in a loop
for text in texts:
    result = coref_bridge.from_text(text)
    # Do something with the result
    # ...
    
    # Explicitly delete large objects when done
    del result

# When completely done, delete the bridge to free model memory
del coref_bridge
```

### Batch Processing and Parallel Execution

For efficient processing of multiple documents or large datasets:

```python
import concurrent.futures
from bridgenlp.adapters.hf_sentiment import HuggingFaceSentimentBridge

# Create a bridge adapter
sentiment_bridge = HuggingFaceSentimentBridge()

# Process multiple documents in parallel
texts = [
    "I love this product!",
    "This was a terrible experience.",
    "The service was okay but could be better.",
    # ... more texts
]

# Method 1: Using the CLI with parallel processing
# bridgenlp predict --model sentiment --file texts.txt --parallel --max-workers 4

# Method 2: Manual parallel processing
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(sentiment_bridge.from_text, texts))

# Process results
for text, result in zip(texts, results):
    sentiment = result.roles[0]['label']
    score = result.roles[0]['score']
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (confidence: {score:.2f})")
```

### Multilingual Support

BridgeNLP provides comprehensive support for multilingual text processing through its enhanced token alignment capabilities:

```python
from bridgenlp.aligner import TokenAligner
from bridgenlp.adapters.hf_translation import HuggingFaceTranslationBridge

# Create a token aligner with multilingual capabilities
aligner = TokenAligner()

# Detect script type automatically
text_en = "This is a test document."
text_zh = "这是一个测试文档。"
text_ar = "هذا مستند اختبار."
text_ru = "Это тестовый документ."

# Automatic script detection
script_en = aligner._detect_script_type(text_en)  # "latin"
script_zh = aligner._detect_script_type(text_zh)  # "cjk"
script_ar = aligner._detect_script_type(text_ar)  # "arabic"
script_ru = aligner._detect_script_type(text_ru)  # "cyrillic"

# Script-specific tokenization
tokens_en = aligner._tokenize_latin(text_en)
tokens_zh = aligner._tokenize_cjk(text_zh)  # Character-based tokenization
tokens_ar = aligner._tokenize_arabic(text_ar)
tokens_ru = aligner._tokenize_latin(text_ru)  # Cyrillic uses latin-like tokenization

# Translation with alignment information
translator = HuggingFaceTranslationBridge(
    model_name="Helsinki-NLP/opus-mt-en-zh",
    source_lang="en",
    target_lang="zh"
)

# Get translation with alignment information
result = translator.from_text("This is a multilingual test document.")
translation = result.roles[0]["text"]
alignment_info = result.roles[0]["alignment"]

# Access alignment matrix between source and target tokens
for align in alignment_info["alignments"]:
    source_indices = align["source_indices"]
    target_indices = align["target_indices"]
    score = align["score"]
    source_tokens = [alignment_info["source_tokens"][i] for i in source_indices]
    target_tokens = [alignment_info["target_tokens"][i] for i in target_indices]
    print(f"Source: {' '.join(source_tokens)} → Target: {''.join(target_tokens)} (score: {score:.2f})")
```

### Handling Large Documents

For large documents, use the optimized methods:

```python
import spacy
from bridgenlp.adapters.spacy_ner import SpacyNERBridge

# Load a lightweight spaCy model
nlp = spacy.load("en_core_web_sm")

# Create a bridge adapter
ner_bridge = SpacyNERBridge()

# Process a large document
with open("large_document.txt", "r") as f:
    text = f.read()

# Method 1: Use the optimized token aligner for large documents
# The aligner will automatically use optimized methods for large documents

# Method 2: Process in batches for better memory efficiency
batch_size = 10000  # characters
for i in range(0, len(text), batch_size):
    batch = text[i:i+batch_size]
    result = ner_bridge.from_text(batch)
    # Process the batch results
    # ...

# Method 3: Use the CLI with batch processing
# bridgenlp predict --model ner --file large_document.txt --batch-size 100
```

## Performance Notes and Limitations

- **Memory usage**: BridgeNLP is designed to minimize memory usage by avoiding deep copies and cleaning up resources.
- **First-run latency**: The first call to a bridge adapter will load the underlying model, which may take time.
- **Token alignment**: The token aligner does its best to map between different tokenization schemes, but may not be perfect in all cases.
- **Model dependencies**: Each adapter has its own dependencies, which are only loaded when the adapter is used.

## Memory Safety Principles

BridgeNLP follows these principles to ensure memory safety:

1. **Lazy loading**: Models are only loaded when needed.
2. **Resource cleanup**: Models are properly unloaded when no longer needed.
3. **Minimal copying**: Data is passed by reference where possible.
4. **Explicit caching**: Caching is opt-in and controlled.
5. **No global state**: State is contained within objects.

## Development

```bash
# Clone the repository
git clone https://github.com/dcondrey/bridgenlp.git
cd bridgenlp

# Install development dependencies
pip install -e ".[allennlp,huggingface]"
pip install pytest pytest-cov mypy black isort build twine

# Run tests
pytest

# Check coverage
pytest --cov=bridgenlp

# Format code
black bridgenlp tests examples
isort bridgenlp tests examples
```

## License

MIT License
