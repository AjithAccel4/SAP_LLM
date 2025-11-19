"""
Real Model Loader Utility for Integration Tests.

Manages loading, caching, and cleanup of real ML models for integration tests.
Handles GPU memory management, model quantization, and proper cleanup.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
import yaml

import torch
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class RealModelLoader:
    """
    Load and manage real ML models for integration tests.

    Features:
    - Automatic model downloading and caching
    - GPU memory management
    - Model quantization (4-bit, 8-bit)
    - Proper cleanup and resource management
    - Multiple model instances

    Usage:
        loader = RealModelLoader("config/models.yaml")
        vision_model = loader.load_vision_encoder()
        language_model = loader.load_language_decoder()
        reasoning_model = loader.load_reasoning_engine()

        # Use models...

        loader.cleanup()  # Free all GPU memory
    """

    def __init__(
        self,
        config_path: str = "config/models.yaml",
        use_quantization: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the model loader.

        Args:
            config_path: Path to models.yaml configuration
            use_quantization: Whether to use quantized models (saves memory)
            device: Device to load models on (auto-detected if None)
        """
        self.config_path = Path(config_path)
        self.use_quantization = use_quantization

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load configuration
        self.config = self._load_config()

        # Track loaded models for cleanup
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_processors: Dict[str, Any] = {}

        # Memory tracking
        self.initial_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0

        logger.info(f"RealModelLoader initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Quantization: {self.use_quantization}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU memory: {self.initial_memory:.2f} GB free")

    def _load_config(self) -> Dict[str, Any]:
        """Load models configuration from YAML."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file not found."""
        return {
            "vision_encoder": {
                "model": "microsoft/layoutlmv3-base",
                "quantization": None,
            },
            "language_decoder": {
                "model": "meta-llama/Llama-2-7b-hf",
                "quantization": "8bit",
            },
            "reasoning_engine": {
                "model": "mistralai/Mixtral-8x7B-v0.1",
                "quantization": "4bit",
            },
        }

    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        return total - allocated

    def _get_quantization_config(self, bits: Optional[str]) -> Optional[BitsAndBytesConfig]:
        """
        Get quantization configuration.

        Args:
            bits: '4bit', '8bit', or None

        Returns:
            BitsAndBytesConfig or None
        """
        if not self.use_quantization or bits is None:
            return None

        if bits == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif bits == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            return None

    def load_vision_encoder(
        self,
        force_reload: bool = False,
    ) -> tuple:
        """
        Load LayoutLMv3 vision encoder model.

        Args:
            force_reload: Force reload even if already cached

        Returns:
            Tuple of (model, processor)
        """
        if "vision_encoder" in self.loaded_models and not force_reload:
            logger.info("Returning cached vision encoder")
            return (
                self.loaded_models["vision_encoder"],
                self.loaded_processors["vision_encoder"],
            )

        config = self.config.get("vision_encoder", {})
        model_name = config.get("model", "microsoft/layoutlmv3-base")

        logger.info(f"Loading vision encoder: {model_name}")
        start_time = time.time()

        try:
            # Load processor
            processor = LayoutLMv3Processor.from_pretrained(
                model_name,
                apply_ocr=False,  # We provide our own OCR
            )

            # Load model
            quantization_config = self._get_quantization_config(
                config.get("quantization")
            )

            model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
            )

            if self.device != "cuda":
                model = model.to(self.device)

            model.eval()  # Set to evaluation mode

            # Cache
            self.loaded_models["vision_encoder"] = model
            self.loaded_processors["vision_encoder"] = processor

            load_time = time.time() - start_time
            logger.info(f"Vision encoder loaded in {load_time:.2f}s")

            if torch.cuda.is_available():
                logger.info(f"  GPU memory: {self._get_gpu_memory():.2f} GB free")

            return model, processor

        except Exception as e:
            logger.error(f"Failed to load vision encoder: {e}")
            raise

    def load_language_decoder(
        self,
        force_reload: bool = False,
    ) -> tuple:
        """
        Load LLaMA-2 language decoder model.

        Args:
            force_reload: Force reload even if already cached

        Returns:
            Tuple of (model, tokenizer)
        """
        if "language_decoder" in self.loaded_models and not force_reload:
            logger.info("Returning cached language decoder")
            return (
                self.loaded_models["language_decoder"],
                self.loaded_processors["language_decoder"],
            )

        config = self.config.get("language_decoder", {})
        model_name = config.get("model", "meta-llama/Llama-2-7b-hf")

        logger.info(f"Loading language decoder: {model_name}")
        start_time = time.time()

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
            )

            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with quantization
            quantization_config = self._get_quantization_config(
                config.get("quantization", "8bit")
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.device != "cuda" and quantization_config is None:
                model = model.to(self.device)

            model.eval()

            # Cache
            self.loaded_models["language_decoder"] = model
            self.loaded_processors["language_decoder"] = tokenizer

            load_time = time.time() - start_time
            logger.info(f"Language decoder loaded in {load_time:.2f}s")

            if torch.cuda.is_available():
                logger.info(f"  GPU memory: {self._get_gpu_memory():.2f} GB free")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load language decoder: {e}")
            raise

    def load_reasoning_engine(
        self,
        force_reload: bool = False,
    ) -> tuple:
        """
        Load Mixtral reasoning engine model.

        Args:
            force_reload: Force reload even if already cached

        Returns:
            Tuple of (model, tokenizer)
        """
        if "reasoning_engine" in self.loaded_models and not force_reload:
            logger.info("Returning cached reasoning engine")
            return (
                self.loaded_models["reasoning_engine"],
                self.loaded_processors["reasoning_engine"],
            )

        config = self.config.get("reasoning_engine", {})
        model_name = config.get("model", "mistralai/Mixtral-8x7B-v0.1")

        logger.info(f"Loading reasoning engine: {model_name}")
        start_time = time.time()

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with quantization (required for Mixtral due to size)
            quantization_config = self._get_quantization_config(
                config.get("quantization", "4bit")
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.device != "cuda" and quantization_config is None:
                model = model.to(self.device)

            model.eval()

            # Cache
            self.loaded_models["reasoning_engine"] = model
            self.loaded_processors["reasoning_engine"] = tokenizer

            load_time = time.time() - start_time
            logger.info(f"Reasoning engine loaded in {load_time:.2f}s")

            if torch.cuda.is_available():
                logger.info(f"  GPU memory: {self._get_gpu_memory():.2f} GB free")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load reasoning engine: {e}")
            raise

    def load_all_models(self) -> Dict[str, tuple]:
        """
        Load all models at once.

        Returns:
            Dictionary with all models and processors
        """
        logger.info("Loading all models...")

        models = {}

        try:
            models["vision"] = self.load_vision_encoder()
            models["language"] = self.load_language_decoder()
            models["reasoning"] = self.load_reasoning_engine()

            logger.info("All models loaded successfully")
            return models

        except Exception as e:
            logger.error(f"Failed to load all models: {e}")
            self.cleanup()
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "loaded_models": list(self.loaded_models.keys()),
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            info["gpu_memory_free_gb"] = self._get_gpu_memory()

        # Model parameters
        for name, model in self.loaded_models.items():
            info[f"{name}_parameters"] = sum(
                p.numel() for p in model.parameters()
            )

        return info

    def cleanup(self):
        """Clean up all loaded models and free GPU memory."""
        logger.info("Cleaning up models...")

        # Delete models
        for name in list(self.loaded_models.keys()):
            logger.info(f"  Deleting {name}")
            del self.loaded_models[name]

        # Delete processors
        for name in list(self.loaded_processors.keys()):
            del self.loaded_processors[name]

        # Clear caches
        self.loaded_models.clear()
        self.loaded_processors.clear()

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            final_memory = self._get_gpu_memory()
            freed_memory = final_memory - self.initial_memory

            logger.info(f"  GPU memory freed: {freed_memory:.2f} GB")
            logger.info(f"  GPU memory available: {final_memory:.2f} GB")

        logger.info("Cleanup complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()
        return False


# Convenience functions for tests
def create_test_model_loader(
    config_path: str = "config/models.yaml",
    use_quantization: bool = True,
) -> RealModelLoader:
    """
    Create a model loader for tests with sensible defaults.

    Args:
        config_path: Path to model configuration
        use_quantization: Use quantized models (recommended for tests)

    Returns:
        RealModelLoader instance
    """
    return RealModelLoader(
        config_path=config_path,
        use_quantization=use_quantization,
    )


def get_model_cache_dir() -> Path:
    """Get the model cache directory."""
    cache_dir = os.environ.get("HF_CACHE_DIR", "/models/huggingface_cache")
    return Path(cache_dir)


def check_models_downloaded() -> Dict[str, bool]:
    """
    Check if models are already downloaded.

    Returns:
        Dictionary mapping model names to download status
    """
    cache_dir = get_model_cache_dir()

    models = {
        "layoutlmv3": "models--microsoft--layoutlmv3-base",
        "llama2": "models--meta-llama--Llama-2-7b-hf",
        "mixtral": "models--mistralai--Mixtral-8x7B-v0.1",
    }

    status = {}
    for name, model_dir in models.items():
        model_path = cache_dir / model_dir
        status[name] = model_path.exists() and any(model_path.iterdir())

    return status
