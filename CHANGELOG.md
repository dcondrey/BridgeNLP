# Changelog

All notable changes to BridgeNLP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Memory leak in TokenAligner for large documents**:
  - Limited the LRU cache size for `_normalize_text` and added bypass for long inputs
  - Optimized `_find_promising_regions` to use less memory with large documents
  - Rewrote `_fuzzy_align_large_doc` with memory-efficient data structures and algorithms
  - Enhanced `_calculate_similarity_score` to avoid creating large temporary data structures
  - Fixed bug in `MockDoc.__getitem__` using `stop` instead of `end` for slices
  - Added proper `__getitem__` implementation to `MockSpan` to make it subscriptable
- **Improved JSON serialization in BridgeResult**:
  - Enhanced handling of special tokens like control characters and null bytes
  - Added proper serialization of numpy arrays and numpy scalar values
  - Implemented sanitization of problematic string content
  - Added support for objects with .text attributes (like spaCy spans)
  - Improved handling of custom objects with proper string conversion
  - Fixed handling of nested data structures with non-serializable items
  - Added comprehensive test suite for special token handling
- **Resolved thread safety issues in pipeline processing**:
  - Added proper locking mechanisms for shared resources
  - Implemented thread-safe condition checking
  - Enhanced thread-safe cache operations with LRU tracking
  - Fixed race conditions in pipeline execution
  - Improved result combining when processing spaCy documents
  - Added thread-safe extension registration and value handling
  - Created comprehensive concurrent test suite
- **Improved spaCy model handling**:
  - Added automatic download of spaCy models when not found
  - Implemented fallback to blank models when downloads fail
  - Improved error handling in spaCy initialization
  - Enhanced recovery strategies when models are unavailable
  - Fixed misleading warning messages about missing models
  - Updated to use full model names instead of language codes
- Thread safety and garbage collection improvements in token alignment operations
- Improved efficiency in token normalization for large documents

### Added
- Memory-optimized version of text normalization for large inputs
- Smart token sampling for large documents to improve memory efficiency
- Size limits on temporary data structures to prevent unbounded memory growth

### Changed
- Made TokenAligner more resilient to different document sizes
- Optimized indexing and slicing operations for better memory usage
- Refactored core alignment algorithms to use tiered approaches based on document size

## [0.3.0] - 2023-05-10

### Added

- **Text Generation Capabilities**: Added new adapters for text generation tasks
  - Text summarization adapter using BART and other models
  - Text paraphrasing adapter with control over variations
  - Translation adapter with support for multiple language pairs
  - Integration with Hugging Face's text generation pipelines

- **Text Generation CLI Support**:
  - Command-line interface support for summarization, paraphrasing, and translation
  - Configuration options for controlling generation parameters

- **New Example**:
  - Added `text_generation_demo.py` showcasing summarization, paraphrasing, and translation
  - Demonstrates complex pipeline with summarization and translation combined

## [0.2.0] - 2023-05-05

### Added

- **Pipeline Composition Framework**: New `Pipeline` class that allows combining multiple adapters into a single pipeline
  - Automatic result combination between different adapters
  - Efficient token alignment between adapter outputs
  - Result caching to improve performance for repeated calls
  - Performance metrics collection and reporting
  - Thread-safe operation with proper resource management
  
- **Enhanced Configuration System**:
  - Added pipeline-specific configuration options
  - Support for result caching and cache size configuration
  - Pipeline parallelization options
  - Made `model_type` optional to support pipeline configurations

- **New Examples**:
  - Added `pipeline_demo.py` showcasing how to use the pipeline composition framework
  - Combined NER and sentiment analysis in a single pipeline example

### Changed

- Updated configuration defaults for better out-of-the-box performance
- Made API more flexible with improved type hints
- Improved documentation with pipeline examples
- Bumped version number to 0.2.0 to reflect the significant feature addition

### Fixed

- Fixed memory management for large document processing
- Improved error handling for invalid configurations

## [0.1.3] - 2023-04-15

### Added

- Added JSON serialization for complex data types
- Improved thread safety across adapter implementations

### Fixed

- Fixed handling of non-serializable spans and tuples in JSON serialization

## [0.1.2] - 2023-04-10

### Added

- Automatic spaCy model download for NER adapter
- Improved error handling for missing models

## [0.1.1] - 2023-04-05

### Changed

- Optimized TokenAligner for better performance
- Added thread safety to core components
- Improved memory management for large documents

## [0.1.0] - 2023-04-01

### Added

- Initial release of BridgeNLP
- Base adapter framework
- Token alignment utilities
- spaCy pipeline integration
- Support for:
  - Coreference resolution (AllenNLP)
  - Named Entity Recognition (spaCy)
  - Semantic Role Labeling (Hugging Face)
  - Sentiment Analysis (Hugging Face)
  - Text Classification (Hugging Face)
  - Question Answering (Hugging Face)
  - NLTK integration