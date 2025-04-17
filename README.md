# BridgeNLP

A minimal, robust, universal NLP model-to-pipeline integration framework.

## What is BridgeNLP?

BridgeNLP serves as a **universal adapter layer** between advanced NLP models (e.g., AllenNLP, Hugging Face) and structured token pipelines (e.g., spaCy). Its core goal is to allow developers to integrate models like coreference resolution, semantic role labeling, or named entity recognition into token-based applications in a clean, aligned, and memory-safe manner.

Key features:
- **Minimal dependencies**: Only requires spaCy and NumPy by default
- **Modular and extensible**: Add only the model adapters you need
- **Token alignment**: Seamlessly map between different tokenization schemes
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

For Hugging Face support (semantic role labeling):
```bash
pip install bridgenlp[huggingface]
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

BridgeNLP provides several adapters for different NLP tasks:

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

## Advanced Usage

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
