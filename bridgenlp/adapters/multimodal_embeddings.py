"""
Bridge adapter for multimodal embeddings using CLIP and similar models.

This adapter generates embeddings for text, images, or text-image pairs
for similarity, retrieval, and classification tasks.
"""

import os
from typing import List, Optional, Union, Any, Dict, Tuple

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

from ..config import BridgeConfig
from ..multimodal_base import MultimodalBridgeBase
from ..result import BridgeResult
from ..utils import (
    configure_device, 
    get_param_with_fallback, 
    get_or_create_model,
    create_model_key,
    validate_text_input
)


class MultimodalEmbeddingsBridge(MultimodalBridgeBase):
    """
    Bridge adapter for multimodal embeddings using models like CLIP.
    
    This adapter integrates vision-language models for generating joint
    embeddings of text and images.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 device: Union[int, str] = -1, config: Optional[BridgeConfig] = None):
        """
        Initialize the bridge adapter.
        
        Args:
            model_name: Name or path of the Hugging Face model to use
            device: Device to run the model on (-1 for CPU, >=0 for specific GPU, or "cuda"/"cpu")
            config: Configuration for the adapter
        """
        # Always call the parent constructor first
        super().__init__(config)
        
        # Store model name, using config if provided
        self.model_name = get_param_with_fallback(
            model_name, config, "model_name", default_value="openai/clip-vit-base-patch32"
        )
        
        # Extract configuration options with defaults
        self.device = get_param_with_fallback(device, config, "device", default_value=-1)
        self.device_idx = configure_device(self.device)
        
        # Additional parameters from config
        self.normalize = get_param_with_fallback(True, config, "params", "normalize", True)
        
        # Initialize model components lazily
        self._model = None
        self._processor = None
        self._model_loaded = False
        
        # Create a unique key for this model in the registry
        self.model_key = create_model_key(self.model_name, "multimodal_embeddings", self.device_idx)
    
    def _load_model_and_processor(self):
        """
        Lazily load the multimodal embedding model and processor.
        
        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            
            # Define a creator function for shared model registry
            def create_model_components():
                # CLIP is the main model type supported currently
                processor = CLIPProcessor.from_pretrained(self.model_name)
                model = CLIPModel.from_pretrained(self.model_name)
                
                # Move to device
                if (self.device_idx >= 0 and torch.cuda.is_available()):
                    model = model.to(f"cuda:{self.device_idx}")
                
                # Set to evaluation mode
                model.eval()
                
                return {
                    "model": model,
                    "processor": processor
                }
            
            # Get or create model components
            components = get_or_create_model(
                self.model_key, 
                create_model_components
            )
            
            # Assign components
            self._model = components["model"]
            self._processor = components["processor"]
            
            self._model_loaded = True
            
        except ImportError:
            raise ImportError(
                "Required dependencies not installed for multimodal embeddings. "
                "Install with: pip install 'bridgenlp[multimodal]' or "
                "manually install: transformers>=4.25.0 torch>=1.10.0 pillow>=9.0.0"
            )
    
    def _process_text_and_image(self, text: str, image_path: str):
        """
        Process text and image together.
        
        Args:
            text: Text to process
            image_path: Path to the image file
            
        Returns:
            Model inputs
            
        Raises:
            ImportError: If PIL is not installed
            ValueError: If the inputs cannot be processed
        """
        try:
            from PIL import Image
            import torch
            
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model_and_processor()
            
            # Validate text input
            text = validate_text_input(text)
            
            # Open image
            image = Image.open(image_path).convert("RGB")
            
            # Process text and image
            inputs = self._processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            # Move to device if needed
            if self.device_idx >= 0 and torch.cuda.is_available():
                inputs = {k: v.to(f"cuda:{self.device_idx}") for k, v in inputs.items()}
            
            return inputs
                
        except ImportError:
            raise ImportError("PIL not installed. Install with: pip install 'bridgenlp[multimodal]' or manually install: pillow>=9.0.0")
        except Exception as e:
            raise ValueError(f"Error processing inputs: {str(e)}")
    
    def from_text_and_image(self, text: str, image_path: str) -> BridgeResult:
        """
        Process text and image together and return structured results.
        
        Args:
            text: Text to process
            image_path: Path to the image file
            
        Returns:
            BridgeResult containing the processed information
        """
        with self._measure_performance():
            # Validate image path
            validated_path = self.validate_image_path(image_path)
            
            # Validate text input
            text = validate_text_input(text)
            
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model_and_processor()
            
            # Process text and image
            inputs = self._process_text_and_image(text, validated_path)
            
            # Get embeddings
            import torch
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                
                # Extract text and image embeddings
                text_embeds = outputs.text_embeds
                image_embeds = outputs.image_embeds
                
                # Calculate similarity score
                if self.normalize:
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                
                similarity = torch.matmul(text_embeds, image_embeds.T)[0][0].item()
                
                # Convert to numpy arrays
                text_embeds = text_embeds.cpu().numpy()[0]
                image_embeds = image_embeds.cpu().numpy()[0]
            
            # Tokenize text
            tokens = text.split()
            
            # Create image features dictionary
            import os
            image_features = {
                "image_path": os.path.abspath(validated_path),
                "embedding": image_embeds.tolist()
            }
            
            # Store combined embedding
            multimodal_embeddings = ((text_embeds + image_embeds) / 2).tolist()
            
            # Create roles with embeddings and similarity info
            roles = [{
                "text_embedding": text_embeds.tolist(),
                "similarity_score": similarity
            }]
            
            # Return structured result
            return BridgeResult(
                tokens=tokens,
                roles=roles,
                image_features=image_features,
                multimodal_embeddings=multimodal_embeddings
            )
    
    def from_image(self, image_path: str) -> BridgeResult:
        """
        Process an image and return embedding results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            BridgeResult containing the processed information
        """
        with self._measure_performance():
            # Validate image path
            validated_path = self.validate_image_path(image_path)
            
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model_and_processor()
            
            try:
                from PIL import Image
                import torch
                
                # Open image
                image = Image.open(validated_path).convert("RGB")
                
                # Process image
                inputs = self._processor(
                    images=[image],
                    return_tensors="pt"
                )
                
                # Move to device if needed
                if self.device_idx >= 0 and torch.cuda.is_available():
                    inputs = {k: v.to(f"cuda:{self.device_idx}") for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self._model.get_image_features(**inputs)
                    
                    # Normalize if requested
                    if self.normalize:
                        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy array
                    image_embeds = outputs.cpu().numpy()[0]
                
                # Create image features dictionary
                import os
                image_features = {
                    "image_path": os.path.abspath(validated_path),
                    "embedding": image_embeds.tolist()
                }
                
                # Generate a simple token for compatibility
                tokens = ["image_embedding"]
                
                # Return structured result
                return BridgeResult(
                    tokens=tokens,
                    image_features=image_features,
                    multimodal_embeddings=image_embeds.tolist()
                )
                
            except ImportError:
                raise ImportError("PIL not installed. Install with: pip install 'bridgenlp[multimodal]' or manually install: pillow>=9.0.0")
            except Exception as e:
                raise ValueError(f"Error processing image: {str(e)}")
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process text and return embedding results.
        
        Args:
            text: Text to process
            
        Returns:
            BridgeResult containing the processed information
        """
        with self._measure_performance():
            # Validate text input
            text = validate_text_input(text)
            
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model_and_processor()
            
            import torch
            
            # Process text
            inputs = self._processor(
                text=[text],
                return_tensors="pt",
                padding=True
            )
            
            # Move to device if needed
            if self.device_idx >= 0 and torch.cuda.is_available():
                inputs = {k: v.to(f"cuda:{self.device_idx}") for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self._model.get_text_features(**inputs)
                
                # Normalize if requested
                if self.normalize:
                    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                
                # Convert to numpy array
                text_embeds = outputs.cpu().numpy()[0]
            
            # Tokenize text
            tokens = text.split()
            
            # Update token count for metrics
            self._metrics["total_tokens"] += len(tokens)
            
            # Store embeddings in the roles field
            roles = [{"embedding": text_embeds.tolist()}]
            
            # Return structured result
            return BridgeResult(
                tokens=tokens,
                roles=roles,
                multimodal_embeddings=text_embeds.tolist()
            )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text and return embedding results.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing embeddings
        """
        # Reconstruct text from tokens
        text = " ".join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and return an enhanced Doc with embeddings.
        
        This method checks if the Doc has an image_path attribute for multimodal processing.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            The same Doc with additional embedding attributes attached
        """
        with self._measure_performance():
            # Check for custom image path attribute
            if hasattr(doc._, "image_path") and doc._.image_path:
                image_path = doc._.image_path
                text = doc.text
                # Process text and image together
                result = self.from_text_and_image(text, image_path)
            else:
                # Process text only
                text = doc.text
                result = self.from_text(text)
            
            return result.attach_to_spacy(doc)
    
    def from_audio(self, audio_path: str) -> BridgeResult:
        """
        Process an audio file and return structured results.
        
        This adapter doesn't support audio processing, so it returns an error result.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            BridgeResult containing an error message
        """
        # Validate path exists but return an error result
        self.validate_audio_path(audio_path)
        
        # Return a result with a warning
        return BridgeResult(
            tokens=["audio_not_supported"],
            labels=["warning: audio processing not supported by multimodal embeddings model"]
        )
    
    def calculate_similarity(self, text: str, image_path: str) -> float:
        """
        Calculate the similarity score between text and image.
        
        Args:
            text: Text to compare
            image_path: Path to the image file
            
        Returns:
            Similarity score (0 to 1)
        """
        with self._measure_performance():
            # Process text and image
            result = self.from_text_and_image(text, image_path)
            
            # Extract similarity score
            similarity = result.roles[0]["similarity_score"]
            
            return similarity
    
    def cleanup(self):
        """
        Clean up resources used by this adapter.
        
        This method is called when the adapter is used as a context manager
        or when it's garbage collected.
        """
        from ..utils import unload_model, free_memory
        
        # Unload the model if requested in config
        if (hasattr(self, "config") and self.config and 
            hasattr(self.config, "unload_on_del") and self.config.unload_on_del):
            unload_model(self.model_key)
            self._model = None
            self._processor = None
            self._model_loaded = False
            free_memory()