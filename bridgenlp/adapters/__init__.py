"""
Model-specific adapters for BridgeNLP.

This module provides adapters for various NLP models, allowing them to be
used with the BridgeNLP framework. Each adapter is imported conditionally
to avoid hard dependencies.
"""

# Import adapters conditionally to avoid hard dependencies
try:
    from .allen_coref import AllenNLPCorefBridge
except ImportError:
    pass

try:
    from .hf_srl import HuggingFaceSRLBridge
except ImportError:
    pass

try:
    from .spacy_ner import SpacyNERBridge
except ImportError:
    pass

# New adapters
try:
    from .hf_sentiment import HuggingFaceSentimentBridge
except ImportError:
    pass

try:
    from .hf_classification import HuggingFaceClassificationBridge
except ImportError:
    pass

try:
    from .hf_qa import HuggingFaceQABridge
except ImportError:
    pass

# Other frameworks
try:
    from .nltk_adapter import NLTKBridge
except ImportError:
    pass
