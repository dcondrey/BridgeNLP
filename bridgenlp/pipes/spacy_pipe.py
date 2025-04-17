"""
spaCy pipeline component for BridgeNLP.
"""

from typing import Callable, Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from ..base import BridgeBase


class SpacyBridgePipe:
    """
    spaCy pipeline component for BridgeNLP adapters.
    
    This component allows seamless integration of BridgeNLP adapters
    into spaCy pipelines using nlp.add_pipe().
    """
    
    def __init__(self, bridge: BridgeBase, name: str = "bridgenlp"):
        """
        Initialize the spaCy pipeline component.
        
        Args:
            bridge: The BridgeNLP adapter to use
            name: Name for this pipeline component
        """
        self.bridge = bridge
        self.name = name
    
    def __call__(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc with the bridge adapter.
        
        This method is called when the pipeline component is executed.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            The processed Doc with additional attributes
        """
        return self.bridge.from_spacy(doc)


# Register the factory with spaCy
@Language.factory("bridgenlp")
def create_bridge_component(nlp: Language, name: str, bridge: BridgeBase) -> SpacyBridgePipe:
    """
    Factory function for creating a SpacyBridgePipe component.
    
    This allows the component to be added to a spaCy pipeline with:
    nlp.add_pipe("bridgenlp", config={"bridge": bridge})
    
    Args:
        nlp: The spaCy Language object
        name: Name for this pipeline component
        bridge: The BridgeNLP adapter to use
        
    Returns:
        Configured SpacyBridgePipe component
    """
    return SpacyBridgePipe(bridge=bridge, name=name)
