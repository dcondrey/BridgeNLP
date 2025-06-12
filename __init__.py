"""
BridgeNLP: A framework for integrating NLP models with token-based pipelines.

This package provides adapters and utilities for working with various NLP models
and frameworks in a consistent way.
"""

# Import core components
from .bridgenlp.base import BridgeBase
from .bridgenlp.config import BridgeConfig
from .bridgenlp.multimodal_base import MultimodalBridgeBase
from .bridgenlp.pipeline import Pipeline
from .bridgenlp.result import BridgeResult

# Create namespaces
try:
    from .bridgenlp import adapters, pipes
except Exception:  # pragma: no cover - optional deps may fail
    adapters = None
    pipes = None

__all__ = [
    "adapters",
    "pipes",
    "BridgeBase",
    "BridgeConfig",
    "MultimodalBridgeBase",
    "Pipeline",
    "BridgeResult",
]
