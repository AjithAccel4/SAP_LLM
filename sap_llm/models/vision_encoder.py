"""
Vision Encoder for SAP_LLM.

Based on LayoutLMv3 for visual-text feature extraction from documents.
Handles document understanding, layout analysis, and visual feature encoding.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Model,
    LayoutLMv3Processor,
)

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class VisionEncoder(nn.Module):
    """
    Vision encoder based on LayoutLMv3.

    This encoder processes document images along with OCR tokens to extract
    visual-text features for downstream tasks.

    Architecture:
    - Base: LayoutLMv3-base (300M parameters)
    - Input: Document images + OCR tokens + bounding boxes
    - Output: 768-dim embeddings per page region

    Args:
        model_name: HuggingFace model name or path
        num_labels: Number of labels for classification tasks
        device: Device to run model on (cuda/cpu)
        precision: Model precision (fp32/fp16/int8)
    """

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        num_labels: Optional[int] = None,
        device: str = "cuda",
        precision: str = "fp16",
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.precision = precision

        logger.info(f"Initializing VisionEncoder: {model_name}")
        logger.info(f"Device: {device}, Precision: {precision}")

        # Load processor
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name,
            apply_ocr=False,  # We provide OCR externally
        )

        # Load model based on task
        if num_labels is not None:
            # Classification mode
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
            )
            self.task_type = "classification"
        else:
            # Feature extraction mode
            self.model = LayoutLMv3Model.from_pretrained(model_name)
            self.task_type = "feature_extraction"

        # Move to device
        self.model.to(device)

        # Set precision
        if precision == "fp16":
            self.model.half()
        elif precision == "int8":
            # Quantization
            self.model = self._quantize_model(self.model)

        # Set to eval mode
        self.model.eval()

        logger.info(f"VisionEncoder initialized: {self._count_parameters():,} parameters")

    def _count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize model to INT8."""
        from torch.quantization import quantize_dynamic

        logger.info("Quantizing model to INT8...")
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized_model

    def preprocess(
        self,
        image,
        words: List[str],
        boxes: List[List[int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess inputs for the model.

        Args:
            image: PIL Image or numpy array
            words: List of OCR words
            boxes: List of bounding boxes [x1, y1, x2, y2] (normalized 0-1000)

        Returns:
            Dictionary of model inputs
        """
        # Process inputs
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        return encoding

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Image tensor [B, C, H, W]
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            bbox: Bounding boxes [B, seq_len, 4]

        Returns:
            Dictionary with:
            - last_hidden_state: [B, seq_len, hidden_size]
            - pooler_output: [B, hidden_size]
            - logits: [B, num_labels] (if classification mode)
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                **kwargs,
            )

        result = {
            "last_hidden_state": outputs.last_hidden_state,
        }

        if hasattr(outputs, "pooler_output"):
            result["pooler_output"] = outputs.pooler_output

        if hasattr(outputs, "logits"):
            result["logits"] = outputs.logits

        return result

    def encode(
        self,
        image,
        words: List[str],
        boxes: List[List[int]],
    ) -> torch.Tensor:
        """
        Encode document to feature embeddings.

        Args:
            image: Document image
            words: OCR words
            boxes: Bounding boxes

        Returns:
            Embeddings tensor [seq_len, hidden_size]
        """
        # Preprocess
        inputs = self.preprocess(image, words, boxes)

        # Forward pass
        outputs = self.forward(**inputs)

        # Return last hidden state
        return outputs["last_hidden_state"].squeeze(0)

    def classify(
        self,
        image,
        words: List[str],
        boxes: List[List[int]],
    ) -> Tuple[int, float]:
        """
        Classify document type.

        Args:
            image: Document image
            words: OCR words
            boxes: Bounding boxes

        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.task_type != "classification":
            raise ValueError("Model must be in classification mode")

        # Preprocess
        inputs = self.preprocess(image, words, boxes)

        # Forward pass
        outputs = self.forward(**inputs)

        # Get predictions
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_class].item()

        return predicted_class, confidence

    def save(self, output_path: str) -> None:
        """Save model and processor."""
        logger.info(f"Saving VisionEncoder to {output_path}")
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = "cuda",
        precision: str = "fp16",
    ) -> "VisionEncoder":
        """Load model from path."""
        logger.info(f"Loading VisionEncoder from {model_path}")
        return cls(
            model_name=model_path,
            device=device,
            precision=precision,
        )
