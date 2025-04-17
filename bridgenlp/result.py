"""
Result container for NLP model outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.tokens import Doc


@dataclass
class BridgeResult:
    """
    Container for standardized NLP model outputs.
    
    This class provides a consistent interface for different types of
    NLP model outputs, making them compatible with token-based pipelines.
    """
    
    tokens: List[str]
    spans: List[Tuple[int, int]] = field(default_factory=list)
    clusters: List[List[Tuple[int, int]]] = field(default_factory=list)
    roles: List[Dict[str, Any]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the result to a JSON-serializable dictionary.
        
        Returns:
            Dict containing all the result data
        """
        result = {"tokens": self.tokens}
        
        # Only include non-empty fields to reduce size
        if self.spans:
            result["spans"] = self.spans
        if self.clusters:
            result["clusters"] = self.clusters
        if self.roles:
            result["roles"] = self.roles
        if self.labels:
            result["labels"] = self.labels
            
        return result
    
    def attach_to_spacy(self, doc: Doc) -> Doc:
        """
        Attach the result data to a spaCy Doc as custom extensions.
        
        This method registers and assigns Doc._ extensions safely and idempotently.
        
        Args:
            doc: spaCy Doc to attach results to
            
        Returns:
            The same Doc with additional attributes attached
        """
        # Register extensions if they don't exist
        if not Doc.has_extension("nlp_bridge_spans"):
            Doc.set_extension("nlp_bridge_spans", default=None)
        if not Doc.has_extension("nlp_bridge_clusters"):
            Doc.set_extension("nlp_bridge_clusters", default=None)
        if not Doc.has_extension("nlp_bridge_roles"):
            Doc.set_extension("nlp_bridge_roles", default=None)
        if not Doc.has_extension("nlp_bridge_labels"):
            Doc.set_extension("nlp_bridge_labels", default=None)
        
        # Assign values
        doc._.nlp_bridge_spans = self.spans
        doc._.nlp_bridge_clusters = self.clusters
        doc._.nlp_bridge_roles = self.roles
        doc._.nlp_bridge_labels = self.labels
        
        return doc
