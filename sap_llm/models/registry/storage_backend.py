"""
Storage Backend for Model Registry.

Provides abstraction for model artifact storage (filesystem, S3, etc.)
"""

import json
import logging
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for model storage backends."""

    @abstractmethod
    def save_model(
        self,
        model: torch.nn.Module,
        model_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Save model artifacts.

        Args:
            model: PyTorch model
            model_id: Unique model identifier
            metadata: Model metadata

        Returns:
            Path/URI to saved model
        """
        pass

    @abstractmethod
    def load_model(self, model_id: str) -> tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Load model artifacts.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, metadata)
        """
        pass

    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """
        Delete model artifacts.

        Args:
            model_id: Model identifier

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def model_exists(self, model_id: str) -> bool:
        """
        Check if model exists.

        Args:
            model_id: Model identifier

        Returns:
            True if exists
        """
        pass


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend.

    Stores models in a directory structure:
    storage_path/
        {model_type}/
            {model_id}/
                model.pt
                metadata.json
                config.json
    """

    def __init__(self, storage_path: str = "./model_registry"):
        """
        Initialize local storage backend.

        Args:
            storage_path: Root directory for model storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"LocalStorageBackend initialized at {self.storage_path}")

    def _get_model_path(self, model_id: str) -> Path:
        """Get path to model directory."""
        # Extract model type from model_id (format: {type}_{version}_{timestamp})
        model_type = model_id.split('_')[0] if '_' in model_id else 'default'
        return self.storage_path / model_type / model_id

    def save_model(
        self,
        model: torch.nn.Module,
        model_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Save model to local filesystem.

        Args:
            model: PyTorch model
            model_id: Unique model identifier
            metadata: Model metadata

        Returns:
            Path to saved model
        """
        model_path = self._get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save model weights
            model_file = model_path / "model.pt"
            torch.save(model.state_dict(), model_file)
            logger.info(f"Saved model weights to {model_file}")

            # Save metadata
            metadata_file = model_path / "metadata.json"
            metadata_with_timestamp = {
                **metadata,
                "saved_at": datetime.now().isoformat(),
                "model_id": model_id,
                "storage_backend": "local"
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata_with_timestamp, f, indent=2)
            logger.info(f"Saved metadata to {metadata_file}")

            # Save model config if available
            if hasattr(model, 'config'):
                config_file = model_path / "config.json"
                with open(config_file, 'w') as f:
                    json.dump(model.config, f, indent=2)
                logger.info(f"Saved config to {config_file}")

            return str(model_path)

        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            # Cleanup on failure
            if model_path.exists():
                shutil.rmtree(model_path)
            raise

    def load_model(self, model_id: str) -> tuple[Optional[torch.nn.Module], Dict[str, Any]]:
        """
        Load model from local filesystem.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, metadata)
        """
        model_path = self._get_model_path(model_id)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")

        try:
            # Load metadata
            metadata_file = model_path / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            logger.info(f"Loaded metadata for {model_id}")

            # Note: Actual model loading requires model architecture
            # In practice, this would load the state dict and apply to a model instance
            # For now, return None for model and just metadata
            # The caller should instantiate the model and load the state dict

            model_file = model_path / "model.pt"
            if model_file.exists():
                metadata["model_path"] = str(model_file)

            return None, metadata

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def delete_model(self, model_id: str) -> bool:
        """
        Delete model from filesystem.

        Args:
            model_id: Model identifier

        Returns:
            True if successful
        """
        model_path = self._get_model_path(model_id)

        if not model_path.exists():
            logger.warning(f"Model not found for deletion: {model_id}")
            return False

        try:
            shutil.rmtree(model_path)
            logger.info(f"Deleted model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    def model_exists(self, model_id: str) -> bool:
        """
        Check if model exists.

        Args:
            model_id: Model identifier

        Returns:
            True if exists
        """
        model_path = self._get_model_path(model_id)
        return model_path.exists() and (model_path / "model.pt").exists()

    def list_models(self, model_type: Optional[str] = None) -> list[str]:
        """
        List all models in storage.

        Args:
            model_type: Optional filter by model type

        Returns:
            List of model IDs
        """
        models = []

        if model_type:
            type_path = self.storage_path / model_type
            if type_path.exists():
                models = [d.name for d in type_path.iterdir() if d.is_dir()]
        else:
            for type_dir in self.storage_path.iterdir():
                if type_dir.is_dir():
                    models.extend([d.name for d in type_dir.iterdir() if d.is_dir()])

        return models

    def get_storage_size(self, model_id: Optional[str] = None) -> int:
        """
        Get storage size in bytes.

        Args:
            model_id: Optional model ID to get size for specific model

        Returns:
            Size in bytes
        """
        if model_id:
            model_path = self._get_model_path(model_id)
            if not model_path.exists():
                return 0

            return sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        else:
            # Total storage size
            return sum(f.stat().st_size for f in self.storage_path.rglob('*') if f.is_file())
