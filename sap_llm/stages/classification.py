"""
Stage 3: Classification - Document Type Identification

Uses fine-tuned LayoutLMv3 to classify documents into 15+ categories.
"""

from typing import Any, Dict

from sap_llm.models.vision_encoder import VisionEncoder
from sap_llm.stages.base_stage import BaseStage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ClassificationStage(BaseStage):
    """
    Document type classification stage.

    Model: LayoutLMv3-base fine-tuned
    Classes: 15 document types
    Accuracy target: ≥95%
    Latency: <200ms
    """

    DOCUMENT_TYPES = [
        "PURCHASE_ORDER",
        "SUPPLIER_INVOICE",
        "SALES_ORDER",
        "CUSTOMER_INVOICE",
        "GOODS_RECEIPT",
        "ADVANCED_SHIPPING_NOTICE",
        "DELIVERY_NOTE",
        "CREDIT_NOTE",
        "DEBIT_NOTE",
        "PAYMENT_ADVICE",
        "REMITTANCE_ADVICE",
        "STATEMENT_OF_ACCOUNT",
        "QUOTE",
        "CONTRACT",
        "OTHER",
    ]

    def __init__(self, config: Any = None):
        super().__init__(config)

        # Load vision encoder for classification
        self.model = None  # Lazy load
        self.confidence_threshold = (
            getattr(config, "confidence_threshold", 0.90) if config else 0.90
        )

    def _load_model(self):
        """Lazy load classification model."""
        if self.model is None:
            logger.info("Loading classification model...")
            self.model = VisionEncoder(
                num_labels=len(self.DOCUMENT_TYPES),
                device="cuda",
                precision="fp16",
            )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify document type.

        Args:
            input_data: {
                "enhanced_images": List[Image],
                "ocr_results": List[Dict],
            }

        Returns:
            {
                "doc_type": str,
                "confidence": float,
                "class_probabilities": Dict[str, float],
            }
        """
        # Load model
        self._load_model()

        # Get first page (most documents have type on page 1)
        image = input_data["enhanced_images"][0]
        ocr_result = input_data["ocr_results"][0]

        words = ocr_result["words"]
        boxes = ocr_result["boxes"]

        # Classify
        class_idx, confidence = self.model.classify(image, words, boxes)

        doc_type = self.DOCUMENT_TYPES[class_idx]

        logger.info(f"Classification: {doc_type} (confidence: {confidence:.4f})")

        return {
            "doc_type": doc_type,
            "confidence": confidence,
            "class_index": class_idx,
        }
