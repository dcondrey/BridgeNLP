"""
BridgeNLP: A universal adapter layer between NLP models and token pipelines.

This package provides a clean interface for integrating advanced NLP models
(like AllenNLP, Hugging Face) with structured token pipelines (like spaCy).
"""

__version__ = "0.1.3"

# Import commonly used classes for easier access
from .base import BridgeBase
from .result import BridgeResult
from .aligner import TokenAligner
from .config import BridgeConfig
