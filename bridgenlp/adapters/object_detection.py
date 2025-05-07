"""
Bridge adapter for object detection in images using Hugging Face models.

This adapter integrates computer vision models for detecting objects
in images and attaching results to token-based pipelines.
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
    create_model_key
)


class ObjectDetectionBridge(MultimodalBridgeBase):
    """
    Bridge adapter for object detection using Hugging Face models.
    
    This adapter integrates computer vision models for object detection
    with the BridgeNLP framework.
    """
    
    def __init__(self, model_name: str = "facebook/detr-resnet-50", 
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
            model_name, config, "model_name", default_value="facebook/detr-resnet-50"
        )
        
        # Extract configuration options with defaults
        self.device = get_param_with_fallback(device, config, "device", default_value=-1)
        self.device_idx = configure_device(self.device)
        
        # Additional parameters from config
        self.threshold = get_param_with_fallback(0.9, config, "params", "threshold", 0.9)
        
        # For image size, handle both tuple and dict formats
        default_size = (800, 800)  # (height, width)
        if config and config.image_size:
            if isinstance(config.image_size, dict):
                self.image_size = (
                    config.image_size.get("height", default_size[0]), 
                    config.image_size.get("width", default_size[1])
                )
            else:
                # Assume it's already a tuple or similar sequence
                try:
                    self.image_size = tuple(config.image_size)[:2]
                except (TypeError, ValueError):
                    self.image_size = default_size
        else:
            self.image_size = default_size
            
        # Initialize model components lazily
        self._model = None
        self._processor = None
        self._model_loaded = False
        
        # Create a unique key for this model in the registry
        self.model_key = create_model_key(self.model_name, "object_detection", self.device_idx)
    
    def _load_model_and_processor(self):
        """
        Lazily load the object detection model and processor.
        
        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
            
            # Define a creator function for shared model registry
            def create_model_components():
                # Load model and processor
                processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                model = AutoModelForObjectDetection.from_pretrained(self.model_name)
                
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
                "Required dependencies not installed for object detection. "
                "Install with: pip install 'bridgenlp[multimodal]' or "
                "manually install: transformers>=4.25.0 torch>=1.10.0 pillow>=9.0.0"
            )
    
    def _preprocess_image(self, image_path: str):
        """
        Preprocess an image for the model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
            
        Raises:
            ImportError: If PIL is not installed
            ValueError: If the image cannot be processed
        """
        try:
            from PIL import Image
            import torch
            
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model_and_processor()
            
            # Open image
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            inputs = self._processor(images=image, return_tensors="pt")
            
            # Move to device if needed
            if self.device_idx >= 0 and torch.cuda.is_available():
                inputs = {k: v.to(f"cuda:{self.device_idx}") for k, v in inputs.items()}
            
            return inputs, image.size
                
        except ImportError:
            raise ImportError("PIL not installed. Install with: pip install 'bridgenlp[multimodal]' or manually install: pillow>=9.0.0")
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def from_image(self, image_path: str) -> BridgeResult:
        """
        Process an image and return object detection results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            BridgeResult containing detected objects
            
        Raises:
            ValueError: If the image path is invalid
        """
        with self._measure_performance():
            # Validate image path
            validated_path = self.validate_image_path(image_path)
            
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model_and_processor()
            
            # Preprocess image
            inputs, original_size = self._preprocess_image(validated_path)
            
            # Run inference
            import torch
            
            with torch.no_grad():
                outputs = self._model(**inputs)
            
            # Post-process results
            # The model returns logits and bounding boxes
            probas = outputs.logits.softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > self.threshold
            
            # Convert kept predictions to the original image size
            target_sizes = torch.tensor([original_size])
            if self.device_idx >= 0 and torch.cuda.is_available():
                target_sizes = target_sizes.to(f"cuda:{self.device_idx}")
                
            boxes = outputs.pred_boxes[0, keep]
            
            # Convert from center format to (x1, y1, x2, y2)
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct
            
            # Get class labels
            prob_values, prob_indices = probas[keep].max(-1)
            
            # Convert to Python lists
            boxes = boxes.cpu().numpy()
            prob_values = prob_values.cpu().numpy()
            prob_indices = prob_indices.cpu().numpy()
            
            # Get object labels from the model's config
            id2label = self._model.config.id2label
            
            # Create list of detected objects
            detected_objects = []
            token_spans = []
            labels = []
            
            for i, (box, prob_idx, prob_val) in enumerate(zip(boxes, prob_indices, prob_values)):
                x1, y1, x2, y2 = box.tolist()
                label = id2label[prob_idx.item()]
                confidence = prob_val.item()
                
                # Round coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Create object entry
                obj = {
                    "id": i,
                    "label": label,
                    "score": confidence,
                    "box": [x1, y1, x2, y2]
                }
                
                detected_objects.append(obj)
                labels.append(label)
                token_spans.append((i, i+1))  # Each object gets a token span
            
            # Create tokens from labels
            tokens = labels
            
            # Create image features dictionary
            import os
            image_features = {
                "image_path": os.path.abspath(validated_path),
                "size": original_size,
                "num_objects": len(detected_objects)
            }
            
            # Generate a simple caption
            if detected_objects:
                caption = f"Image containing {', '.join(labels[:5])}"
                if len(labels) > 5:
                    caption += f" and {len(labels) - 5} other objects"
                captions = [caption]
            else:
                captions = ["No objects detected in the image"]
            
            # Return structured result
            return BridgeResult(
                tokens=tokens,
                spans=token_spans,
                labels=labels,
                captions=captions,
                detected_objects=detected_objects,
                image_features=image_features
            )
    
    def from_text(self, text: str) -> BridgeResult:
        """
        Process raw text and return results.
        
        This adapter primarily works with images, but implements this method
        for compatibility with the base interface.
        
        Args:
            text: Raw text to process (interpreted as an image path)
            
        Returns:
            BridgeResult containing processed information
        """
        # Attempt to interpret the text as an image path
        if os.path.exists(text) and os.path.isfile(text):
            return self.from_image(text)
        else:
            # Return a minimal result with tokens and a warning
            tokens = text.split()
            return BridgeResult(
                tokens=tokens,
                labels=["warning: text-only input to object detection model"]
            )
    
    def from_tokens(self, tokens: List[str]) -> BridgeResult:
        """
        Process pre-tokenized text and return results.
        
        This adapter primarily works with images, but implements this method
        for compatibility with the base interface.
        
        Args:
            tokens: List of pre-tokenized strings
            
        Returns:
            BridgeResult containing processed information
        """
        # Reconstruct text from tokens
        text = " ".join(tokens)
        return self.from_text(text)
    
    def from_spacy(self, doc: Doc) -> Doc:
        """
        Process a spaCy Doc and return an enhanced Doc with detected objects.
        
        This method assumes the Doc has a custom attribute with the image path.
        If not, it falls back to using the Doc text as a potential image path.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            The same Doc with additional object detection attributes attached
        """
        with self._measure_performance():
            # Check for custom image path attribute
            image_path = None
            if hasattr(doc._, "image_path") and doc._.image_path:
                image_path = doc._.image_path
            else:
                # Try to interpret the text as an image path
                text = doc.text
                if os.path.exists(text) and os.path.isfile(text):
                    image_path = text
            
            # Process the image if found, otherwise just return the original doc
            if image_path:
                result = self.from_image(image_path)
                return result.attach_to_spacy(doc)
            else:
                # Create minimal result with a warning
                tokens = [token.text for token in doc]
                result = BridgeResult(
                    tokens=tokens,
                    labels=["warning: no image path found for object detection"]
                )
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
            labels=["warning: audio processing not supported by object detection model"]
        )
    
    def from_text_and_image(self, text: str, image_path: str) -> BridgeResult:
        """
        Process text and image together and return results.
        
        For this adapter, we ignore the text and just process the image.
        
        Args:
            text: Text to process (ignored in this implementation)
            image_path: Path to the image file
            
        Returns:
            BridgeResult containing the processed information
        """
        # Just process the image and ignore the text
        return self.from_image(image_path)
    
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