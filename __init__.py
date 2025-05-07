"""
BridgeNLP: A framework for integrating NLP models with token-based pipelines.

This package provides adapters and utilities for working with various NLP models
and frameworks in a consistent way.
"""

from bridgenlp import adapters, pipes
from bridgenlp.base import BridgeBase
from bridgenlp.config import BridgeConfig
from bridgenlp.multimodal_base import MultimodalBridgeBase
from bridgenlp.pipeline import Pipeline
from bridgenlp.result import BridgeResult

__all__ = [
    "adapters",
    "pipes",
    "BridgeBase",
    "BridgeConfig",
    "MultimodalBridgeBase",
    "Pipeline",
    "BridgeResult",
]
