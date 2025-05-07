"""
BridgeNLP: A universal adapter layer between AI models and structured pipelines.

This package provides a clean interface for integrating advanced NLP and multimodal models
(like AllenNLP, Hugging Face) with structured pipelines (like spaCy), supporting both
text and multimodal (image, audio) inputs.
"""

__version__ = "0.4.0"

# Import commonly used classes for easier access
from .base import BridgeBase
from .result import BridgeResult
from .aligner import TokenAligner
from .config import BridgeConfig
from .pipeline import Pipeline
from .multimodal_base import MultimodalBridgeBase

# Import multimodal adapters
# These imports are optional and will fail gracefully if dependencies are missing
try:
    from .adapters.image_captioning import ImageCaptioningBridge
except ImportError:
    pass

try:
    from .adapters.object_detection import ObjectDetectionBridge
except ImportError:
    pass

try:
    from .adapters.multimodal_embeddings import MultimodalEmbeddingsBridge
except ImportError:
    pass
