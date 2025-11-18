"""
Vision Encoder for SAP_LLM.

Based on LayoutLMv3 for visual-text feature extraction from documents.
Handles document understanding, layout analysis, and visual feature encoding.

This module provides:
1. Multi-task learning with document classification and field extraction
2. Support for 15 document types and 35 PO subtypes
3. Token classification for 180+ SAP fields
4. ONNX export for optimized inference
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Model,
    LayoutLMv3Processor,
    LayoutLMv3PreTrainedModel,
)

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


# Document type and subtype labels
DOCUMENT_TYPES = [
    "INVOICE",
    "PURCHASE_ORDER",
    "SALES_ORDER",
    "GOODS_RECEIPT",
    "DELIVERY_NOTE",
    "CREDIT_MEMO",
    "DEBIT_MEMO",
    "PAYMENT_ADVICE",
    "REMITTANCE_ADVICE",
    "STATEMENT",
    "CONTRACT",
    "QUOTATION",
    "RFQ",
    "ASN",
    "OTHER",
]

PO_SUBTYPES = [
    "STANDARD",
    "BLANKET",
    "CONTRACT",
    "SCHEDULING_AGREEMENT",
    "SERVICE",
    "CONSIGNMENT",
    "SUBCONTRACTING",
    "THIRD_PARTY",
    "STOCK_TRANSPORT",
    "FRAMEWORK",
    "RELEASE_ORDER",
    "RUSH_ORDER",
    "EMERGENCY_ORDER",
    "PLANT_ORDER",
    "MAINTENANCE_ORDER",
    # Additional 20 subtypes
    "CAPITAL_EXPENDITURE",
    "OPERATING_EXPENDITURE",
    "RECURRING_ORDER",
    "ONE_TIME_ORDER",
    "MULTI_YEAR_CONTRACT",
    "ANNUAL_CONTRACT",
    "QUARTERLY_CONTRACT",
    "MASTER_AGREEMENT",
    "CALL_OFF_ORDER",
    "DISTRIBUTION_ORDER",
    "TRANSFER_ORDER",
    "INTER_COMPANY_ORDER",
    "CROSS_COMPANY_ORDER",
    "EXTERNAL_PROCUREMENT",
    "INTERNAL_PROCUREMENT",
    "DIRECT_PROCUREMENT",
    "INDIRECT_PROCUREMENT",
    "MRO_ORDER",
    "SPARE_PARTS_ORDER",
    "RAW_MATERIAL_ORDER",
]


class MultiTaskLayoutLMv3(LayoutLMv3PreTrainedModel):
    """
    Multi-task LayoutLMv3 model for SAP document understanding.

    Supports three tasks simultaneously:
    1. Document type classification (15 classes)
    2. PO subtype classification (35 classes)
    3. Token classification for field extraction (180+ fields)

    Args:
        config: LayoutLMv3Config
        num_doc_types: Number of document type classes (default: 15)
        num_po_subtypes: Number of PO subtype classes (default: 35)
        num_token_labels: Number of token classification labels (default: 180)
    """

    def __init__(
        self,
        config: LayoutLMv3Config,
        num_doc_types: int = 15,
        num_po_subtypes: int = 35,
        num_token_labels: int = 180,
    ):
        super().__init__(config)

        self.num_doc_types = num_doc_types
        self.num_po_subtypes = num_po_subtypes
        self.num_token_labels = num_token_labels

        # Base LayoutLMv3 model
        self.layoutlmv3 = LayoutLMv3Model(config)

        # Classification heads
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Document type classification head
        self.doc_type_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_doc_types),
        )

        # PO subtype classification head
        self.po_subtype_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_po_subtypes),
        )

        # Token classification head for field extraction
        self.token_classifier = nn.Linear(config.hidden_size, num_token_labels)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        doc_type_labels: Optional[torch.Tensor] = None,
        po_subtype_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs.

        Returns:
            Dictionary containing:
            - doc_type_logits: [batch_size, num_doc_types]
            - po_subtype_logits: [batch_size, num_po_subtypes]
            - token_logits: [batch_size, seq_len, num_token_labels]
            - loss: Combined loss (if labels provided)
            - doc_type_loss: Document type classification loss
            - po_subtype_loss: PO subtype classification loss
            - token_loss: Token classification loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get base model outputs
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Get [CLS] token representation for classification
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)

        # Document type classification
        doc_type_logits = self.doc_type_classifier(cls_output)

        # PO subtype classification
        po_subtype_logits = self.po_subtype_classifier(cls_output)

        # Token classification for field extraction
        sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)

        # Calculate losses if labels are provided
        total_loss = None
        doc_type_loss = None
        po_subtype_loss = None
        token_loss = None

        if doc_type_labels is not None or po_subtype_labels is not None or token_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            total_loss = 0.0

            # Document type classification loss
            if doc_type_labels is not None:
                doc_type_loss = loss_fct(
                    doc_type_logits.view(-1, self.num_doc_types),
                    doc_type_labels.view(-1),
                )
                total_loss += doc_type_loss

            # PO subtype classification loss
            if po_subtype_labels is not None:
                po_subtype_loss = loss_fct(
                    po_subtype_logits.view(-1, self.num_po_subtypes),
                    po_subtype_labels.view(-1),
                )
                total_loss += po_subtype_loss

            # Token classification loss
            if token_labels is not None:
                # Only compute loss on non-ignored tokens
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.num_token_labels)[active_loss]
                active_labels = token_labels.view(-1)[active_loss]
                token_loss = loss_fct(active_logits, active_labels)
                total_loss += token_loss

        return {
            "loss": total_loss,
            "doc_type_loss": doc_type_loss,
            "po_subtype_loss": po_subtype_loss,
            "token_loss": token_loss,
            "doc_type_logits": doc_type_logits,
            "po_subtype_logits": po_subtype_logits,
            "token_logits": token_logits,
            "last_hidden_state": sequence_output,
            "attentions": outputs.attentions if output_attentions else None,
        }


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


def export_to_onnx(
    model: Union[MultiTaskLayoutLMv3, VisionEncoder],
    output_path: str,
    sample_input: Optional[Dict[str, torch.Tensor]] = None,
    opset_version: int = 14,
) -> None:
    """
    Export model to ONNX format for optimized inference.

    Args:
        model: Model to export (MultiTaskLayoutLMv3 or VisionEncoder)
        output_path: Path to save ONNX model
        sample_input: Sample input for tracing (if None, will create dummy input)
        opset_version: ONNX opset version (default: 14)
    """
    import onnx
    import onnxruntime

    logger.info(f"Exporting model to ONNX: {output_path}")

    # Prepare model for export
    if isinstance(model, VisionEncoder):
        model_to_export = model.model
    else:
        model_to_export = model

    model_to_export.eval()

    # Create dummy input if not provided
    if sample_input is None:
        batch_size = 1
        seq_length = 512
        image_size = 224

        sample_input = {
            "input_ids": torch.randint(0, 30522, (batch_size, seq_length)),
            "bbox": torch.randint(0, 1000, (batch_size, seq_length, 4)),
            "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long),
            "pixel_values": torch.randn(batch_size, 3, image_size, image_size),
        }

    # Move to CPU for ONNX export
    model_to_export = model_to_export.cpu()
    sample_input = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sample_input.items()}

    # Dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "bbox": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "pixel_values": {0: "batch_size"},
    }

    # Output names depend on model type
    if isinstance(model_to_export, MultiTaskLayoutLMv3):
        output_names = ["doc_type_logits", "po_subtype_logits", "token_logits"]
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}
    else:
        output_names = ["logits"]
        dynamic_axes["logits"] = {0: "batch_size"}

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model_to_export,
            (sample_input,),
            output_path,
            input_names=list(sample_input.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    logger.info(f"ONNX model exported and verified: {output_path}")

    # Test with ONNX Runtime
    try:
        ort_session = onnxruntime.InferenceSession(output_path)
        ort_inputs = {k: v.numpy() for k, v in sample_input.items()}
        ort_outputs = ort_session.run(None, ort_inputs)
        logger.info(f"ONNX Runtime test successful. Output shapes: {[o.shape for o in ort_outputs]}")
    except Exception as e:
        logger.warning(f"ONNX Runtime test failed: {e}")


def benchmark_model(
    model: Union[MultiTaskLayoutLMv3, VisionEncoder],
    sample_input: Dict[str, torch.Tensor],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark model inference latency.

    Args:
        model: Model to benchmark
        sample_input: Sample input for inference
        num_iterations: Number of iterations for benchmarking
        warmup_iterations: Number of warmup iterations

    Returns:
        Dictionary with latency statistics (mean, std, min, max)
    """
    import time
    import numpy as np

    logger.info("Benchmarking model inference latency...")

    # Prepare model
    if isinstance(model, VisionEncoder):
        model_to_bench = model.model
    else:
        model_to_bench = model

    model_to_bench.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_to_bench = model_to_bench.to(device)
    sample_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_input.items()}

    # Warmup
    logger.info(f"Warmup: {warmup_iterations} iterations")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model_to_bench(**sample_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Benchmark
    logger.info(f"Benchmarking: {num_iterations} iterations")
    latencies = []

    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model_to_bench(**sample_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    results = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }

    logger.info(f"Benchmark results: mean={results['mean_ms']:.2f}ms, p95={results['p95_ms']:.2f}ms")

    return results
