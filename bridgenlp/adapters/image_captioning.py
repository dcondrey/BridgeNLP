"""
Bridge adapter for image captioning using Hugging Face models.

This adapter integrates vision-language models for generating captions
from images and attaching them to token-based pipelines. It supports
prompt conditioning to guide caption generation.
"""

import os
from typing import List, Optional, Union, Any, Dict, Literal

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


class ImageCaptioningBridge(MultimodalBridgeBase):
    """
    Bridge adapter for image captioning using Hugging Face models.
    
    This adapter integrates vision-language models for image captioning
    with the BridgeNLP framework. It supports prompt conditioning to guide
    caption generation with different strategies.
    """
    
    def __init__(self, model_name: str = "nlpconnect/vit-gpt2-image-captioning", 
                 device: Union[int, str] = -1, config: Optional[BridgeConfig] = None,
                 enable_prompt_conditioning: bool = False,
                 prompt_strategy: Literal["prefix", "template", "instruction"] = "prefix",
                 default_prompt: str = "Describe this image:",
                 prompt_template: str = "{prompt} {caption}"):
        """
        Initialize the bridge adapter.
        
        Args:
            model_name: Name or path of the Hugging Face model to use
            device: Device to run the model on (-1 for CPU, >=0 for specific GPU, or "cuda"/"cpu")
            config: Configuration for the adapter
            enable_prompt_conditioning: Whether to use prompt conditioning for caption generation
            prompt_strategy: Strategy for applying prompts ("prefix", "template", or "instruction")
            default_prompt: Default prompt to use when none is provided
            prompt_template: Template string for the "template" strategy (use {prompt} and {caption})
        """
        # Always call the parent constructor first
        super().__init__(config)
        
        # Store model name, using config if provided
        self.model_name = get_param_with_fallback(
            model_name, config, "model_name", default_value="nlpconnect/vit-gpt2-image-captioning"
        )
        
        # Extract configuration options with defaults
        self.device = get_param_with_fallback(device, config, "device", default_value=-1)
        self.device_idx = configure_device(self.device)
        
        # Additional parameters from config
        self.max_length = get_param_with_fallback(None, config, "max_length", default_value=32)
        self.num_captions = get_param_with_fallback(1, config, "params", "num_captions", 1)
        
        # For image size, handle both tuple and dict formats
        default_size = (224, 224)  # (height, width)
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
            
        # Image processor name
        self.image_processor_name = get_param_with_fallback(
            None, config, "image_processor", default_value=None
        )
        
        # Prompt conditioning parameters
        self.enable_prompt_conditioning = get_param_with_fallback(
            enable_prompt_conditioning, config, "params", "enable_prompt_conditioning", 
            default_value=False
        )
        
        # Prompt strategy
        valid_strategies = ["prefix", "template", "instruction"]
        prompt_strategy_value = get_param_with_fallback(
            prompt_strategy, config, "params", "prompt_strategy", 
            default_value="prefix"
        )
        if prompt_strategy_value not in valid_strategies:
            prompt_strategy_value = "prefix"
        self.prompt_strategy = prompt_strategy_value
        
        # Default prompt
        self.default_prompt = get_param_with_fallback(
            default_prompt, config, "params", "default_prompt", 
            default_value="Describe this image:"
        )
        
        # Prompt template for the "template" strategy
        self.prompt_template = get_param_with_fallback(
            prompt_template, config, "params", "prompt_template", 
            default_value="{prompt} {caption}"
        )
        
        # The current prompt (can be updated during usage)
        self.current_prompt = self.default_prompt
        
        # Initialize model components lazily
        self._model = None
        self._processor = None
        self._tokenizer = None
        
        # Create a unique key for this model in the registry
        self.model_key = create_model_key(self.model_name, "image_captioning", self.device_idx)
    
    def _load_model_and_processor(self):
        """
        Lazily load the image captioning model and processor.
        
        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            import torch
            from transformers import (
                AutoProcessor, 
                AutoModelForCausalLM, 
                AutoTokenizer,
                VisionEncoderDecoderModel,
                AutoImageProcessor
            )
            
            # Define a creator function for shared model registry
            def create_model_components():
                components = {}
                
                # There are different model types with different interfaces
                if "vit-gpt" in self.model_name:
                    # For VisionEncoderDecoder models like vit-gpt2
                    model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                    processor = AutoImageProcessor.from_pretrained(self.model_name)
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    components["model"] = model
                    components["processor"] = processor
                    components["tokenizer"] = tokenizer
                    components["model_type"] = "vit-gpt"
                else:
                    # For models with a unified processor
                    processor = AutoProcessor.from_pretrained(self.model_name)
                    model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    components["model"] = model
                    components["processor"] = processor
                    components["model_type"] = "processor-based"
                
                # Move to device
                if (self.device_idx >= 0 and torch.cuda.is_available()):
                    components["model"] = components["model"].to(f"cuda:{self.device_idx}")
                
                # Set to evaluation mode
                components["model"].eval()
                
                return components
            
            # Get or create model components
            components = get_or_create_model(
                self.model_key, 
                create_model_components
            )
            
            # Assign components
            self._model = components["model"]
            self._processor = components["processor"]
            if "tokenizer" in components:
                self._tokenizer = components["tokenizer"]
            self._model_type = components["model_type"]
            
            self._model_loaded = True
            
        except ImportError:
            raise ImportError(
                "Required dependencies not installed for image captioning. "
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
            
            # Open and convert image
            image = Image.open(image_path).convert("RGB")
            
            # Process image based on model type
            if self._model_type == "vit-gpt":
                # For VisionEncoderDecoder models
                pixel_values = self._processor(image, return_tensors="pt").pixel_values
                
                # Move to device if needed
                if self.device_idx >= 0 and torch.cuda.is_available():
                    pixel_values = pixel_values.to(f"cuda:{self.device_idx}")
                
                return pixel_values
            else:
                # For processor-based models
                inputs = self._processor(images=image, return_tensors="pt")
                
                # Move to device if needed
                if self.device_idx >= 0 and torch.cuda.is_available():
                    inputs = {k: v.to(f"cuda:{self.device_idx}") for k, v in inputs.items()}
                
                return inputs
                
        except ImportError:
            raise ImportError("PIL not installed. Install with: pip install 'bridgenlp[multimodal]' or manually install: pillow>=9.0.0")
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def from_image(self, image_path: str, prompt: Optional[str] = None) -> BridgeResult:
        """
        Process an image and return captioning results.
        
        Args:
            image_path: Path to the image file
            prompt: Optional text prompt to condition the caption generation
            
        Returns:
            BridgeResult containing captions
            
        Raises:
            ValueError: If the image path is invalid
        """
        with self._measure_performance():
            # Validate image path
            validated_path = self.validate_image_path(image_path)
            
            # If prompt is provided, set it as the current prompt
            if prompt is not None:
                self.set_prompt(prompt)
                
            # For instruction-based prompting, we need to modify generation parameters
            # based on the prompt before model generation
            generation_kwargs = {}
            if self.enable_prompt_conditioning and self.prompt_strategy == "instruction":
                # Some models support direct prompt conditioning
                prompt_text = self.current_prompt
                if "gpt" in self.model_name.lower() or "opt" in self.model_name.lower():
                    # For GPT/OPT style models, add prompt as a prefix to the generation
                    generation_kwargs["prefix"] = prompt_text
                elif "t5" in self.model_name.lower() or "bart" in self.model_name.lower():
                    # For T5/BART style models, condition with encoder prefix
                    generation_kwargs["encoder_prompt"] = prompt_text
            
            # Ensure model is loaded
            if not self._model_loaded:
                self._load_model_and_processor()
            
            # Preprocess image
            inputs = self._preprocess_image(validated_path)
            
            # Generate captions
            if self._model_type == "vit-gpt":
                import torch
                
                with torch.no_grad():
                    # For VisionEncoderDecoder models
                    outputs = self._model.generate(
                        inputs,
                        max_length=self.max_length,
                        num_return_sequences=self.num_captions,
                        do_sample=self.num_captions > 1,  # Use sampling for multiple captions
                        temperature=0.7 if self.num_captions > 1 else 1.0,
                        top_p=0.9 if self.num_captions > 1 else 1.0,
                        **generation_kwargs
                    )
                    
                    # Decode captions
                    captions = []
                    for output in outputs:
                        caption = self._tokenizer.decode(output, skip_special_tokens=True)
                        # Apply post-generation prompt conditioning for non-instruction strategies
                        if self.enable_prompt_conditioning and self.prompt_strategy != "instruction":
                            caption = self._apply_prompt_conditioning(caption)
                        captions.append(caption)
            else:
                import torch
                
                # For processor-based models
                # Combine inputs with generation kwargs
                generation_inputs = {**inputs}
                
                with torch.no_grad():
                    outputs = self._model.generate(
                        **generation_inputs,
                        max_length=self.max_length,
                        num_return_sequences=self.num_captions,
                        do_sample=self.num_captions > 1,
                        **generation_kwargs
                    )
                    
                    # Decode captions (model-specific)
                    captions = []
                    for output in outputs:
                        if hasattr(self._processor, "decode"):
                            caption = self._processor.decode(output, skip_special_tokens=True)
                        elif hasattr(self._processor, "tokenizer") and hasattr(self._processor.tokenizer, "decode"):
                            caption = self._processor.tokenizer.decode(output, skip_special_tokens=True)
                        else:
                            caption = output.cpu().numpy().tolist()
                            
                        # Apply post-generation prompt conditioning for non-instruction strategies
                        if self.enable_prompt_conditioning and self.prompt_strategy != "instruction":
                            caption = self._apply_prompt_conditioning(caption)
                        captions.append(caption)
            
            # Store prompt in the result
            prompt_used = None
            if self.enable_prompt_conditioning:
                prompt_used = self.current_prompt
                
            # Tokenize the first caption for token representation
            tokens = captions[0].split() if captions else []
            
            # Create detected_objects placeholder (for compatibility with object detection)
            detected_objects = []
            
            # Create image features dictionary with basic image info
            import os
            image_features = {
                "image_path": os.path.abspath(validated_path),
                "caption": captions[0] if captions else "",
            }
            
            # Add prompt information to image features if used
            if prompt_used:
                image_features["prompt"] = prompt_used
                image_features["prompt_strategy"] = self.prompt_strategy
            
            # Return structured result
            return BridgeResult(
                tokens=tokens,
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
                labels=["warning: text-only input to image captioning model"]
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
        Process a spaCy Doc and return an enhanced Doc with image captions.
        
        This method assumes the Doc has a custom attribute with the image path.
        If not, it falls back to using the Doc text as a potential image path.
        
        Args:
            doc: spaCy Doc object to process
            
        Returns:
            The same Doc with additional caption attributes attached
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
                    labels=["warning: no image path found for captioning"]
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
            labels=["warning: audio processing not supported by image captioning model"]
        )
    
    def from_text_and_image(self, text: str, image_path: str) -> BridgeResult:
        """
        Process text and image together and return results.
        
        When prompt conditioning is enabled, the text is used as a prompt
        to guide caption generation. Otherwise, it behaves like from_image.
        
        Args:
            text: Text to use as prompt for conditioning caption generation
            image_path: Path to the image file
            
        Returns:
            BridgeResult containing the processed information
        """
        # Use the text as a prompt if prompt conditioning is enabled
        # Otherwise, the prompt will be ignored by from_image
        return self.from_image(image_path, prompt=text)
    
    def set_prompt(self, prompt: str) -> None:
        """
        Set the prompt for conditioning caption generation.
        
        Args:
            prompt: The prompt text to use for guiding caption generation
        """
        self.current_prompt = prompt if prompt else self.default_prompt
        
    def reset_prompt(self) -> None:
        """
        Reset the prompt to the default value.
        """
        self.current_prompt = self.default_prompt
        
    def _apply_prompt_conditioning(self, caption: str, prompt: Optional[str] = None) -> str:
        """
        Apply prompt conditioning to the generated caption based on the selected strategy.
        
        Args:
            caption: The generated caption to condition
            prompt: Optional prompt to use (uses self.current_prompt if None)
            
        Returns:
            The conditioned caption
        """
        if not self.enable_prompt_conditioning:
            return caption
            
        # Use the provided prompt or the current one
        active_prompt = prompt if prompt is not None else self.current_prompt
        
        # Apply conditioning based on strategy
        if self.prompt_strategy == "prefix":
            # Simple prefix strategy: "Prompt text: Caption text"
            return f"{active_prompt} {caption}"
            
        elif self.prompt_strategy == "template":
            # Template-based strategy using the prompt_template
            try:
                return self.prompt_template.format(prompt=active_prompt, caption=caption)
            except KeyError:
                # Fall back to simple strategy if template is invalid
                return f"{active_prompt} {caption}"
                
        elif self.prompt_strategy == "instruction":
            # Instruction-based strategy
            # For many vision-language models, the prompt serves as an instruction
            # that influences generation but isn't part of the output.
            # The caption is returned as-is since the prompt already influenced generation.
            return caption
            
        # Default fallback
        return caption
    
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
            self._tokenizer = None
            self._model_loaded = False
            free_memory()