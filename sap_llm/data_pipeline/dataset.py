"""
PyTorch Dataset for SAP_LLM document training.

Handles:
- Document loading and preprocessing
- Multimodal inputs (image + text)
- Data augmentation
- Efficient caching
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class SAP_LLM_Dataset(Dataset):
    """
    Dataset for SAP_LLM multimodal document understanding.

    Features:
    - Lazy loading of documents
    - Image preprocessing and augmentation
    - Support for multiple document types
    - Caching for performance
    """

    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 image_size: Tuple[int, int] = (224, 224),
                 max_length: int = 512,
                 augment: bool = False,
                 cache_size: int = 1000):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing processed data
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (H, W)
            max_length: Maximum sequence length
            augment: Enable data augmentation
            cache_size: Number of samples to cache in memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_length = max_length
        self.augment = augment
        self.cache_size = cache_size

        # Load metadata
        metadata_file = self.data_dir / split / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        self.samples = metadata.get("documents", [])
        self.num_samples = len(self.samples)

        # In-memory cache
        self.cache = {}

        logger.info(
            f"SAP_LLM_Dataset initialized: split={split}, samples={self.num_samples}"
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
            - input_ids: Tokenized text
            - attention_mask: Attention mask
            - pixel_values: Preprocessed image tensor
            - labels: Ground truth labels
        """
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        # Load sample
        sample_meta = self.samples[idx]

        # Load image
        image_path = sample_meta.get("image_path") or sample_meta.get("stored_path")
        image = self._load_image(image_path)

        # Load text/annotations
        text = sample_meta.get("text", "")
        labels = sample_meta.get("labels", {})

        # Preprocess
        processed_sample = {
            "pixel_values": self._preprocess_image(image),
            "input_ids": self._tokenize_text(text),
            "labels": self._prepare_labels(labels),
            "document_type": sample_meta.get("document_type", "unknown"),
        }

        # Add to cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[idx] = processed_sample

        return processed_sample

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return blank image as fallback
            return Image.new('RGB', self.image_size, (255, 255, 255))

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: PIL Image

        Returns:
            Preprocessed image tensor (C, H, W)
        """
        # Resize
        image = image.resize(self.image_size, Image.BILINEAR)

        # Data augmentation (if training)
        if self.augment and self.split == "train":
            image = self._augment_image(image)

        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply data augmentation.

        Augmentations:
        - Random rotation (-5 to +5 degrees)
        - Random brightness/contrast
        - Random noise
        """
        import random
        from PIL import ImageEnhance

        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor=(255, 255, 255))

        # Random brightness
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Random contrast
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        return image

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """
        Tokenize text input.

        Args:
            text: Input text

        Returns:
            Tokenized text tensor
        """
        # Placeholder - in real implementation, use actual tokenizer
        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
        # tokens = tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True)
        # return torch.tensor(tokens['input_ids'])

        # For now, return dummy tensor
        return torch.zeros(self.max_length, dtype=torch.long)

    def _prepare_labels(self, labels: Dict[str, Any]) -> torch.Tensor:
        """
        Prepare labels for training.

        Args:
            labels: Label dictionary

        Returns:
            Label tensor
        """
        # Placeholder - convert labels to tensor format
        # In real implementation, handle different label types:
        # - Classification labels
        # - Extraction labels (token-level)
        # - Structure labels (bounding boxes, etc.)

        return torch.zeros(self.max_length, dtype=torch.long)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_samples": self.num_samples,
            "split": self.split,
            "document_types": {},
        }

        # Count by document type
        for sample in self.samples:
            doc_type = sample.get("document_type", "unknown")
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1

        return stats


# Collate function for DataLoader
def sap_llm_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching SAP_LLM samples.

    Args:
        batch: List of samples

    Returns:
        Batched tensors
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": (input_ids != 0).long(),
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create dataset
    dataset = SAP_LLM_Dataset(
        data_dir="./data/processed",
        split="train",
        image_size=(224, 224),
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset statistics: {dataset.get_statistics()}")

    # Get sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['pixel_values'].shape}")
