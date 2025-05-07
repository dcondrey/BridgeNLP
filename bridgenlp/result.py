"""
Result container for NLP and multimodal model outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

try:
    import numpy as np
except ImportError:
    np = None

try:
    import spacy
    from spacy.tokens import Doc
except ImportError:
    spacy = None
    Doc = Any


@dataclass
class BridgeResult:
    """
    Container for standardized NLP and multimodal model outputs.
    
    This class provides a consistent interface for different types of
    NLP and multimodal model outputs, making them compatible with token-based pipelines.
    """
    
    tokens: List[str]
    spans: List[Tuple[int, int]] = field(default_factory=list)
    clusters: List[List[Tuple[int, int]]] = field(default_factory=list)
    roles: List[Dict[str, Any]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    
    # New fields for multimodal data
    image_features: Optional[Dict[str, Any]] = None
    audio_features: Optional[Dict[str, Any]] = None
    multimodal_embeddings: Optional[List[float]] = None
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    captions: List[str] = field(default_factory=list)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the result to a JSON-serializable dictionary.
        
        This method handles various special cases, including:
        - Converting tuples to lists for JSON compatibility
        - Converting numpy arrays to lists
        - Converting custom objects to strings
        - Properly handling special characters and control codes
        
        Returns:
            Dict containing all the result data in a JSON-serializable format
        """
        # Create a copy of tokens and convert non-serializable tokens to strings
        tokens = self._sanitize_tokens(self.tokens)
        result = {"tokens": tokens}
        
        # Only include non-empty fields to reduce size
        if self.spans:
            # Convert tuples to lists for JSON serialization
            result["spans"] = [list(span) for span in self.spans]
        if self.clusters:
            # Convert nested tuples to lists for JSON serialization
            result["clusters"] = [[list(span) for span in cluster] for cluster in self.clusters]
        if self.roles:
            # Create a deep copy to avoid modifying the original
            roles = self._deep_copy_roles(self.roles)
            result["roles"] = roles
        if self.labels:
            # Ensure labels are strings
            result["labels"] = [str(label) if not isinstance(label, str) else label 
                               for label in self.labels]
            
        # Add multimodal fields with special handling
        if self.image_features:
            result["image_features"] = self._convert_features_to_serializable(self.image_features)
        if self.audio_features:
            result["audio_features"] = self._convert_features_to_serializable(self.audio_features)
        if self.multimodal_embeddings:
            result["multimodal_embeddings"] = self._convert_embeddings_to_serializable(
                self.multimodal_embeddings
            )
        if self.detected_objects:
            # Create a deep copy to avoid modifying the original
            objects = []
            for obj in self.detected_objects:
                obj_copy = dict(obj)
                for k, v in obj_copy.items():
                    if not self._is_json_serializable(v):
                        obj_copy[k] = str(v)
                objects.append(obj_copy)
            result["detected_objects"] = objects
        if self.captions:
            # Ensure captions are strings
            result["captions"] = [str(caption) if not isinstance(caption, str) else caption 
                                 for caption in self.captions]
            
        # Ensure all values are JSON serializable
        self._ensure_serializable(result)
            
        return result
        
    def _sanitize_tokens(self, tokens: List[Any]) -> List[str]:
        """
        Convert all tokens to JSON-serializable strings.
        
        Args:
            tokens: List of token objects
            
        Returns:
            List of string tokens with special characters properly handled
        """
        result = []
        for token in tokens:
            # Handle different token types
            if isinstance(token, str):
                # String tokens can be used as-is
                result.append(token)
            elif np is not None and isinstance(token, np.ndarray):
                # Convert numpy arrays to lists
                result.append(token.tolist())
            elif hasattr(token, 'text'):
                # Handle spaCy-like objects with .text attribute
                result.append(token.text)
            else:
                # Convert other types to string
                result.append(str(token))
                
        return result
        
    def _deep_copy_roles(self, roles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a deep copy of roles list with serializable values.
        
        Args:
            roles: List of role dictionaries
            
        Returns:
            Copy of roles with all values converted to serializable format
        """
        result = []
        for role in roles:
            role_copy = {}
            for key, value in role.items():
                if isinstance(value, str):
                    # Pass strings through directly
                    role_copy[key] = value
                elif np is not None and isinstance(value, np.ndarray):
                    # Convert numpy arrays to lists
                    role_copy[key] = value.tolist()
                elif hasattr(value, 'text'):
                    # Handle spaCy-like objects with .text attribute
                    role_copy[key] = value.text
                elif not self._is_json_serializable(value):
                    # Convert other non-serializable types to string
                    role_copy[key] = str(value)
                else:
                    # Pass through serializable values
                    role_copy[key] = value
            result.append(role_copy)
        return result
        
    def _convert_features_to_serializable(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert feature dictionary to JSON-serializable format.
        
        Args:
            features: Dictionary of feature data
            
        Returns:
            Copy of features with all values converted to serializable format
        """
        result = {}
        for key, value in features.items():
            if np is not None and isinstance(value, np.ndarray):
                # Convert numpy arrays to lists
                result[key] = value.tolist()
            elif isinstance(value, dict):
                # Recursively convert nested dictionaries
                result[key] = self._convert_features_to_serializable(value)
            elif isinstance(value, list):
                # Handle lists that might contain non-serializable items
                result[key] = self._convert_list_to_serializable(value)
            elif not self._is_json_serializable(value):
                # Convert other non-serializable types to string
                result[key] = str(value)
            else:
                # Pass through serializable values
                result[key] = value
        return result
        
    def _convert_list_to_serializable(self, items: List[Any]) -> List[Any]:
        """
        Convert a list to contain only JSON-serializable items.
        
        Args:
            items: List of items
            
        Returns:
            List with all items converted to serializable format
        """
        result = []
        for item in items:
            if np is not None and isinstance(item, np.ndarray):
                # Convert numpy arrays to lists
                result.append(item.tolist())
            elif isinstance(item, dict):
                # Recursively convert nested dictionaries
                result.append(self._convert_features_to_serializable(item))
            elif isinstance(item, list):
                # Recursively convert nested lists
                result.append(self._convert_list_to_serializable(item))
            elif isinstance(item, tuple):
                # Convert tuples to lists
                result.append(list(item))
            elif not self._is_json_serializable(item):
                # Convert other non-serializable types to string
                result.append(str(item))
            else:
                # Pass through serializable values
                result.append(item)
        return result
        
    def _convert_embeddings_to_serializable(self, embeddings: Any) -> List[float]:
        """
        Convert embeddings to a JSON-serializable list of floats.
        
        Args:
            embeddings: Embedding data (numpy array, list, etc.)
            
        Returns:
            List of float values
        """
        if np is not None and isinstance(embeddings, np.ndarray):
            # Convert numpy array directly to list
            return embeddings.tolist()
        elif isinstance(embeddings, list):
            # Process each item in the list
            result = []
            for item in embeddings:
                if np is not None and isinstance(item, np.ndarray):
                    # Convert numpy values to Python native types
                    result.append(item.tolist())
                elif isinstance(item, (int, float)):
                    # Native numeric types can pass through
                    result.append(item)
                else:
                    # Convert other types to floats if possible, or strings
                    try:
                        result.append(float(item))
                    except (ValueError, TypeError):
                        result.append(float(0))  # Use default value for non-convertible items
            return result
        else:
            # For other types, try to convert to list or use an empty list
            try:
                return list(embeddings)
            except (TypeError, ValueError):
                return []
        
    def _ensure_serializable(self, obj: Union[Dict, List, Any]) -> None:
        """
        Recursively ensure all values in a dictionary or list are JSON serializable.
        
        This is a final validation step to guarantee that the object graph is entirely
        serializable. It handles:
        - Converting tuples to lists
        - Converting non-serializable objects to strings
        - Special handling for numpy arrays and other common types
        - Sanitizing problematic string content
        
        Args:
            obj: Dictionary, list, or value to check
        """
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if isinstance(value, (dict, list)):
                    self._ensure_serializable(value)
                elif isinstance(value, tuple):
                    # Convert tuples to lists for JSON serialization
                    obj[key] = list(value)
                    self._ensure_serializable(obj[key])
                elif np is not None and isinstance(value, np.ndarray):
                    # Convert numpy arrays to lists
                    obj[key] = value.tolist()
                    # Ensure the list elements are also serializable
                    self._ensure_serializable(obj[key])
                elif not self._is_json_serializable(value):
                    # For debugging purposes, warn when non-serializable items are converted
                    warnings.warn(f"Converting non-serializable value for key '{key}' to string")
                    obj[key] = str(value)
                elif isinstance(value, str):
                    # Ensure all strings are valid unicode
                    obj[key] = self._sanitize_string(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, tuple):
                    # Convert tuples to lists for JSON serialization
                    # but don't warn since this is expected
                    obj[i] = list(item)
                    # Check if the tuple contents are serializable
                    self._ensure_serializable(obj[i])
                elif isinstance(item, (dict, list)):
                    self._ensure_serializable(item)
                elif np is not None and isinstance(item, np.ndarray):
                    # Convert numpy arrays to lists
                    obj[i] = item.tolist()
                    # Ensure the list elements are also serializable
                    self._ensure_serializable(obj[i])
                elif not self._is_json_serializable(item):
                    warnings.warn(f"Converting non-serializable value at index {i} to string")
                    obj[i] = str(item)
                elif isinstance(item, str):
                    # Ensure all strings are valid unicode
                    obj[i] = self._sanitize_string(item)
    
    def _sanitize_string(self, s: str) -> str:
        """
        Sanitize a string to ensure it's valid for JSON.
        
        Handles:
        - Control characters that are not valid in JSON
        - Non-UTF-8 encodable characters
        - JSON special characters that may cause issues
        
        Args:
            s: String to sanitize
            
        Returns:
            Sanitized string
        """
        if not s:
            return s
            
        try:
            # Test if string is valid UTF-8 and can be encoded
            s.encode('utf-8')
            
            # Replace JSON-invalid control characters
            result = ""
            for c in s:
                # C0 control characters (except allowed ones in JSON)
                if 0 <= ord(c) <= 0x1F and c not in ('\n', '\r', '\t'):
                    # Replace control characters with their escaped unicode representation
                    result += f"\\u{ord(c):04x}"
                # Non-printable DELETE character
                elif ord(c) == 0x7F:
                    result += "\\u007f"
                else:
                    result += c
                    
            return result
        except UnicodeEncodeError:
            # If there are encoding issues, replace problematic characters
            clean_chars = []
            for c in s:
                try:
                    c.encode('utf-8')
                    clean_chars.append(c)
                except UnicodeEncodeError:
                    clean_chars.append('?')  # Replace with a safe character
            return ''.join(clean_chars)
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """
        Check if an object is JSON serializable.
        
        Tests whether an object can be directly serialized to JSON without conversion.
        
        Args:
            obj: Object to check
            
        Returns:
            True if the object is JSON serializable, False otherwise
        """
        # Handle basic JSON types
        if obj is None or isinstance(obj, (bool, int, float)):
            return True
            
        # Handle strings with special validation
        if isinstance(obj, str):
            try:
                # Try to encode as UTF-8 to verify
                obj.encode('utf-8')
                return True
            except UnicodeEncodeError:
                return False
                
        # Lists and dictionaries are serializable if their contents are
        if isinstance(obj, list):
            return all(self._is_json_serializable(item) for item in obj)
            
        if isinstance(obj, dict):
            return (all(isinstance(k, str) for k in obj.keys()) and
                    all(self._is_json_serializable(v) for v in obj.values()))
            
        # Handle numpy arrays (will be converted to lists)
        if np is not None and isinstance(obj, np.ndarray):
            # We'll convert numpy arrays to lists regardless of content
            # This simplifies the logic and avoids issues with boolean arrays
            return True
                
        # Tuples are serializable as lists in JSON if their contents are serializable
        if isinstance(obj, tuple):
            return all(self._is_json_serializable(item) for item in obj)
            
        # Special handling for numpy scalar types
        if np is not None:
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return True
                
        # All other types are not directly serializable
        return False
    
    def attach_to_spacy(self, doc: Doc) -> Doc:
        """
        Attach the result data to a spaCy Doc as custom extensions.
        
        This method registers and assigns Doc._ extensions safely and idempotently.
        
        Args:
            doc: spaCy Doc to attach results to
            
        Returns:
            The same Doc with additional attributes attached
            
        Raises:
            ValueError: If the doc is None
        """
        if doc is None:
            raise ValueError("Cannot attach results to None")
        
        if spacy is None:
            raise ImportError("spaCy not installed. Install with: pip install spacy")
            
        # Register extensions if they don't exist
        if not Doc.has_extension("nlp_bridge_spans"):
            Doc.set_extension("nlp_bridge_spans", default=None)
        if not Doc.has_extension("nlp_bridge_clusters"):
            Doc.set_extension("nlp_bridge_clusters", default=None)
        if not Doc.has_extension("nlp_bridge_roles"):
            Doc.set_extension("nlp_bridge_roles", default=None)
        if not Doc.has_extension("nlp_bridge_labels"):
            Doc.set_extension("nlp_bridge_labels", default=None)
        
        # Register multimodal extensions if they don't exist
        if not Doc.has_extension("nlp_bridge_image_features"):
            Doc.set_extension("nlp_bridge_image_features", default=None)
        if not Doc.has_extension("nlp_bridge_audio_features"):
            Doc.set_extension("nlp_bridge_audio_features", default=None)
        if not Doc.has_extension("nlp_bridge_multimodal_embeddings"):
            Doc.set_extension("nlp_bridge_multimodal_embeddings", default=None)
        if not Doc.has_extension("nlp_bridge_detected_objects"):
            Doc.set_extension("nlp_bridge_detected_objects", default=None)
        if not Doc.has_extension("nlp_bridge_captions"):
            Doc.set_extension("nlp_bridge_captions", default=None)
        
        # Assign values - text fields
        doc._.nlp_bridge_spans = self.spans
        doc._.nlp_bridge_clusters = self.clusters
        doc._.nlp_bridge_roles = self.roles
        doc._.nlp_bridge_labels = self.labels
        
        # Assign values - multimodal fields
        doc._.nlp_bridge_image_features = self.image_features
        doc._.nlp_bridge_audio_features = self.audio_features
        doc._.nlp_bridge_multimodal_embeddings = self.multimodal_embeddings
        doc._.nlp_bridge_detected_objects = self.detected_objects
        doc._.nlp_bridge_captions = self.captions
        
        return doc
