"""
Stage 4: Type Identifier - Document Subtype Classification

Hierarchical classification for 35+ document subtypes.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sap_llm.models.vision_encoder import VisionEncoder
from sap_llm.stages.base_stage import BaseStage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class TypeIdentifierStage(BaseStage):
    """
    Document subtype identification stage.

    Uses hierarchical classification:
    Level 1: Major category (from Stage 3)
    Level 2: Subtype (35+ total)

    Example PO subtypes: Standard, Blanket, Contract, Service, etc.
    """

    # Comprehensive subtype hierarchy for 35+ SAP document subtypes
    SUBTYPES = {
        "PURCHASE_ORDER": [
            "STANDARD",
            "BLANKET",
            "CONTRACT",
            "SERVICE",
            "SUBCONTRACT",
            "CONSIGNMENT",
            "STOCK_TRANSFER",
            "LIMIT",
            "DROP_SHIP",
            "CAPEX",
        ],
        "SUPPLIER_INVOICE": [
            "STANDARD",
            "CREDIT_MEMO",
            "DEBIT_MEMO",
            "PREPAYMENT",
            "DOWN_PAYMENT",
            "RECURRING",
            "PROFORMA",
            "COMMERCIAL",
        ],
        "SALES_ORDER": [
            "STANDARD",
            "RUSH",
            "SCHEDULED",
            "CONSIGNMENT",
            "RETURNS",
            "CREDIT_ONLY",
        ],
        "CUSTOMER_INVOICE": [
            "STANDARD",
            "CREDIT_NOTE",
            "DEBIT_NOTE",
            "PROFORMA",
            "RECURRING",
            "MILESTONE",
        ],
        "GOODS_RECEIPT": [
            "STANDARD",
            "RETURN_TO_VENDOR",
            "TRANSFER_POSTING",
            "OTHER_RECEIPT",
        ],
        "ADVANCED_SHIPPING_NOTICE": [
            "STANDARD",
            "PARTIAL",
            "COMPLETE",
        ],
        "DELIVERY_NOTE": [
            "STANDARD",
            "PARTIAL",
            "COMPLETE",
            "RETURNS",
        ],
        "CREDIT_NOTE": [
            "SUPPLIER_CREDIT",
            "CUSTOMER_CREDIT",
            "GENERAL",
        ],
        "DEBIT_NOTE": [
            "SUPPLIER_DEBIT",
            "CUSTOMER_DEBIT",
            "GENERAL",
        ],
        "PAYMENT_ADVICE": [
            "STANDARD",
            "PARTIAL",
            "ADVANCE",
        ],
        "REMITTANCE_ADVICE": [
            "STANDARD",
            "CONSOLIDATED",
        ],
        "STATEMENT_OF_ACCOUNT": [
            "MONTHLY",
            "QUARTERLY",
            "ANNUAL",
            "ON_DEMAND",
        ],
        "QUOTE": [
            "REQUEST_FOR_QUOTE",
            "SUPPLIER_QUOTE",
            "SALES_QUOTE",
        ],
        "CONTRACT": [
            "PURCHASING",
            "SALES",
            "MASTER_SERVICE_AGREEMENT",
        ],
        "OTHER": [
            "STANDARD",
        ],
    }

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.vision_encoder = None  # Lazy load
        self.classifiers = {}  # One classifier per document type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = (
            getattr(config, "subtype_confidence_threshold", 0.75) if config else 0.75
        )
        self.enable_multi_label = (
            getattr(config, "enable_multi_label", True) if config else True
        )

    def _load_models(self):
        """Lazy load vision encoder and classification heads."""
        if self.vision_encoder is None:
            logger.info("Loading vision encoder for subtype identification...")
            # Load vision encoder in feature extraction mode
            self.vision_encoder = VisionEncoder(
                model_name="microsoft/layoutlmv3-base",
                num_labels=None,  # Feature extraction mode
                device=self.device,
                precision="fp16" if self.device == "cuda" else "fp32",
            )

            # Initialize classification heads for each document type
            self._initialize_classifiers()

    def _initialize_classifiers(self):
        """Initialize hierarchical classification heads for each document type."""
        logger.info("Initializing hierarchical classifiers...")

        for doc_type, subtypes in self.SUBTYPES.items():
            num_subtypes = len(subtypes)

            # Create a simple classification head: 768-dim -> num_subtypes
            classifier = nn.Sequential(
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(384, num_subtypes),
            ).to(self.device)

            # Set to eval mode (in production, these would be loaded from checkpoint)
            classifier.eval()

            self.classifiers[doc_type] = {
                "model": classifier,
                "subtypes": subtypes,
                "num_classes": num_subtypes,
            }

        logger.info(f"Initialized {len(self.classifiers)} hierarchical classifiers")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify document subtype using hierarchical classification.

        Args:
            input_data: {
                "doc_type": str,
                "enhanced_images": List[Image],
                "ocr_results": List[Dict],
            }

        Returns:
            {
                "subtype": str (primary subtype),
                "subtypes": List[str] (multi-label if enabled),
                "confidence": float (confidence for primary subtype),
                "subtype_scores": Dict[str, float] (all subtype probabilities),
            }
        """
        # Load models if not already loaded
        self._load_models()

        doc_type = input_data["doc_type"]

        # Get available subtypes for this document type
        if doc_type not in self.SUBTYPES:
            logger.warning(f"Unknown document type: {doc_type}, defaulting to STANDARD")
            return {
                "subtype": "STANDARD",
                "subtypes": ["STANDARD"],
                "confidence": 1.0,
                "subtype_scores": {"STANDARD": 1.0},
            }

        # Get first page and OCR results
        image = input_data["enhanced_images"][0]
        ocr_result = input_data["ocr_results"][0]

        words = ocr_result.get("words", [])
        boxes = ocr_result.get("boxes", [])

        # Extract features using vision encoder
        features = self._extract_features(image, words, boxes)

        # Perform hierarchical classification
        subtype_probs = self._classify_subtype(doc_type, features)

        # Get results (single-label or multi-label)
        if self.enable_multi_label:
            results = self._multi_label_results(
                doc_type, subtype_probs, self.confidence_threshold
            )
        else:
            results = self._single_label_results(doc_type, subtype_probs)

        logger.info(
            f"Subtype: {results['subtype']} (confidence: {results['confidence']:.4f})"
        )

        return results

    def _extract_features(self, image, words: List[str], boxes: List[List[int]]) -> torch.Tensor:
        """
        Extract visual-text features using vision encoder.

        Args:
            image: Document image
            words: OCR words
            boxes: Bounding boxes

        Returns:
            Feature tensor [hidden_size]
        """
        with torch.no_grad():
            # Encode document
            embeddings = self.vision_encoder.encode(image, words, boxes)

            # Use [CLS] token or mean pooling
            # Here we use mean pooling over all tokens
            features = embeddings.mean(dim=0)  # [hidden_size]

        return features

    def _classify_subtype(
        self, doc_type: str, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Classify document subtype using hierarchical classifier.

        Args:
            doc_type: Document type (from Stage 3)
            features: Extracted features [hidden_size]

        Returns:
            Subtype probabilities [num_subtypes]
        """
        classifier_info = self.classifiers[doc_type]
        classifier = classifier_info["model"]

        with torch.no_grad():
            # Forward pass through classifier head
            logits = classifier(features.unsqueeze(0))  # [1, num_subtypes]

            # Get probabilities
            probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_subtypes]

        return probs

    def _single_label_results(
        self, doc_type: str, subtype_probs: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Generate single-label classification results.

        Args:
            doc_type: Document type
            subtype_probs: Subtype probabilities

        Returns:
            Classification results
        """
        classifier_info = self.classifiers[doc_type]
        subtypes = classifier_info["subtypes"]

        # Get top prediction
        top_idx = torch.argmax(subtype_probs).item()
        top_subtype = subtypes[top_idx]
        top_confidence = subtype_probs[top_idx].item()

        # Build subtype scores dictionary
        subtype_scores = {
            subtypes[i]: subtype_probs[i].item()
            for i in range(len(subtypes))
        }

        return {
            "subtype": top_subtype,
            "subtypes": [top_subtype],
            "confidence": top_confidence,
            "subtype_scores": subtype_scores,
        }

    def _multi_label_results(
        self, doc_type: str, subtype_probs: torch.Tensor, threshold: float
    ) -> Dict[str, Any]:
        """
        Generate multi-label classification results.

        Documents can have multiple subtypes (e.g., a PO can be both CONTRACT and SERVICE).

        Args:
            doc_type: Document type
            subtype_probs: Subtype probabilities
            threshold: Confidence threshold for multi-label

        Returns:
            Classification results with multiple labels
        """
        classifier_info = self.classifiers[doc_type]
        subtypes = classifier_info["subtypes"]

        # Get top prediction
        top_idx = torch.argmax(subtype_probs).item()
        top_subtype = subtypes[top_idx]
        top_confidence = subtype_probs[top_idx].item()

        # Get all subtypes above threshold
        above_threshold = subtype_probs >= threshold
        predicted_subtypes = [
            subtypes[i]
            for i in range(len(subtypes))
            if above_threshold[i].item()
        ]

        # If no predictions above threshold, use top prediction
        if not predicted_subtypes:
            predicted_subtypes = [top_subtype]

        # Build subtype scores dictionary
        subtype_scores = {
            subtypes[i]: subtype_probs[i].item()
            for i in range(len(subtypes))
        }

        return {
            "subtype": top_subtype,  # Primary subtype
            "subtypes": predicted_subtypes,  # All predicted subtypes
            "confidence": top_confidence,
            "subtype_scores": subtype_scores,
        }

    def get_total_subtypes(self) -> int:
        """Get total number of subtypes across all document types."""
        return sum(len(subtypes) for subtypes in self.SUBTYPES.values())

    def get_hierarchy_info(self) -> Dict[str, Any]:
        """
        Get information about the hierarchical classification structure.

        Returns:
            Dictionary with hierarchy statistics
        """
        total_subtypes = self.get_total_subtypes()
        doc_type_count = len(self.SUBTYPES)

        subtype_distribution = {
            doc_type: len(subtypes)
            for doc_type, subtypes in self.SUBTYPES.items()
        }

        return {
            "total_document_types": doc_type_count,
            "total_subtypes": total_subtypes,
            "subtype_distribution": subtype_distribution,
            "multi_label_enabled": self.enable_multi_label,
            "confidence_threshold": self.confidence_threshold,
        }

    def save_classifiers(self, output_dir: str):
        """
        Save trained classification heads to disk.

        Args:
            output_dir: Directory to save models
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for doc_type, classifier_info in self.classifiers.items():
            model_path = os.path.join(output_dir, f"{doc_type}_classifier.pt")
            torch.save(
                {
                    "model_state_dict": classifier_info["model"].state_dict(),
                    "subtypes": classifier_info["subtypes"],
                    "num_classes": classifier_info["num_classes"],
                },
                model_path,
            )
            logger.info(f"Saved {doc_type} classifier to {model_path}")

    def load_classifiers(self, input_dir: str):
        """
        Load trained classification heads from disk.

        Args:
            input_dir: Directory containing saved models
        """
        import os

        for doc_type, subtypes in self.SUBTYPES.items():
            model_path = os.path.join(input_dir, f"{doc_type}_classifier.pt")

            if not os.path.exists(model_path):
                logger.warning(f"No saved model found for {doc_type} at {model_path}")
                continue

            checkpoint = torch.load(model_path, map_location=self.device)

            # Initialize classifier architecture
            num_subtypes = len(subtypes)
            classifier = nn.Sequential(
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(384, num_subtypes),
            ).to(self.device)

            # Load weights
            classifier.load_state_dict(checkpoint["model_state_dict"])
            classifier.eval()

            self.classifiers[doc_type] = {
                "model": classifier,
                "subtypes": checkpoint["subtypes"],
                "num_classes": checkpoint["num_classes"],
            }

            logger.info(f"Loaded {doc_type} classifier from {model_path}")
