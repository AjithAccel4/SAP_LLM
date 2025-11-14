"""
Unified SAP_LLM Model.

Combines Vision Encoder, Language Decoder, and Reasoning Engine into a
single end-to-end model for document processing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sap_llm.models.language_decoder import LanguageDecoder
from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.models.vision_encoder import VisionEncoder
from sap_llm.utils.logger import get_logger
from sap_llm.utils.timer import Timer

logger = get_logger(__name__)


class UnifiedExtractorModel(nn.Module):
    """
    Unified SAP_LLM model combining all components.

    This model orchestrates the three main components:
    1. Vision Encoder: Visual-text feature extraction
    2. Language Decoder: Structured JSON generation
    3. Reasoning Engine: Decision making and routing

    Total Parameters: ~13.8B
    - Vision Encoder: 300M
    - Language Decoder: 7B
    - Reasoning Engine: 6B active (47B total)

    Args:
        config: Configuration object
        device: Device to run models on
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.config = config
        self.device = device

        logger.info("Initializing UnifiedExtractorModel...")

        # Initialize components
        self.vision_encoder: Optional[VisionEncoder] = None
        self.language_decoder: Optional[LanguageDecoder] = None
        self.reasoning_engine: Optional[ReasoningEngine] = None

        # Load models if config provided
        if config is not None:
            self._load_models_from_config(config)

        logger.info("UnifiedExtractorModel initialized")

    def _load_models_from_config(self, config: Any) -> None:
        """Load all models from configuration."""
        # Vision Encoder
        if hasattr(config, 'models') and hasattr(config.models, 'vision_encoder'):
            ve_config = config.models.vision_encoder
            logger.info("Loading Vision Encoder...")
            self.vision_encoder = VisionEncoder(
                model_name=ve_config.name,
                device=ve_config.device,
                precision=ve_config.precision,
            )

        # Language Decoder
        if hasattr(config, 'models') and hasattr(config.models, 'language_decoder'):
            ld_config = config.models.language_decoder
            logger.info("Loading Language Decoder...")
            self.language_decoder = LanguageDecoder(
                model_name=ld_config.name,
                device=ld_config.device,
                precision=ld_config.precision,
                max_length=ld_config.max_length or 2048,
            )

        # Reasoning Engine
        if hasattr(config, 'models') and hasattr(config.models, 'reasoning_engine'):
            re_config = config.models.reasoning_engine
            logger.info("Loading Reasoning Engine...")
            self.reasoning_engine = ReasoningEngine(
                model_name=re_config.name,
                device=re_config.device,
                precision=re_config.precision,
                max_length=re_config.max_length or 4096,
            )

    def set_vision_encoder(self, vision_encoder: VisionEncoder) -> None:
        """Set vision encoder component."""
        self.vision_encoder = vision_encoder

    def set_language_decoder(self, language_decoder: LanguageDecoder) -> None:
        """Set language decoder component."""
        self.language_decoder = language_decoder

    def set_reasoning_engine(self, reasoning_engine: ReasoningEngine) -> None:
        """Set reasoning engine component."""
        self.reasoning_engine = reasoning_engine

    def classify(
        self,
        image,
        ocr_text: str,
        words: List[str],
        boxes: List[List[int]],
    ) -> Tuple[str, str, float]:
        """
        Classify document type and subtype.

        Args:
            image: Document image
            ocr_text: OCR extracted text
            words: OCR words
            boxes: Bounding boxes

        Returns:
            Tuple of (doc_type, subtype, confidence)
        """
        if self.vision_encoder is None:
            raise ValueError("Vision encoder not initialized")

        with Timer("Document Classification"):
            # Stage 3: Classification
            predicted_class, confidence = self.vision_encoder.classify(
                image,
                words,
                boxes,
            )

            # Map class index to document type
            doc_type = self._map_class_to_type(predicted_class)

            # Stage 4: Type Identifier (simplified - would use separate model)
            subtype = self._identify_subtype(doc_type, ocr_text)

            logger.info(
                f"Classification: {doc_type} / {subtype} "
                f"(confidence: {confidence:.4f})"
            )

            return doc_type, subtype, confidence

    def extract(
        self,
        image,
        ocr_text: str,
        words: List[str],
        boxes: List[List[int]],
        doc_type: str,
        schema: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract structured fields from document.

        Args:
            image: Document image
            ocr_text: OCR text
            words: OCR words
            boxes: Bounding boxes
            doc_type: Document type
            schema: ADC schema

        Returns:
            Tuple of (extracted_data, metadata)
        """
        if self.vision_encoder is None or self.language_decoder is None:
            raise ValueError("Required models not initialized")

        with Timer("Field Extraction"):
            # Stage 5: Extraction
            # Get visual features
            visual_features = self.vision_encoder.encode(image, words, boxes)

            # Extract fields
            extracted_data = self.language_decoder.extract_fields(
                ocr_text,
                doc_type,
                schema,
                visual_features=visual_features,
            )

            # Metadata
            metadata = {
                "doc_type": doc_type,
                "schema": schema,
                "num_fields": len(extracted_data),
            }

            logger.info(f"Extracted {len(extracted_data)} fields")

            return extracted_data, metadata

    def route(
        self,
        adc_json: Dict[str, Any],
        doc_type: str,
        api_schemas: List[Dict[str, Any]],
        similar_cases: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make routing decision for SAP API.

        Args:
            adc_json: Extracted document data
            doc_type: Document type
            api_schemas: Available API schemas
            similar_cases: Similar routing decisions from PMG

        Returns:
            Routing decision with endpoint and payload
        """
        if self.reasoning_engine is None:
            raise ValueError("Reasoning engine not initialized")

        with Timer("Routing Decision"):
            # Stage 8: Routing
            decision = self.reasoning_engine.decide_routing(
                adc_json,
                doc_type,
                api_schemas,
                similar_cases,
            )

            logger.info(
                f"Routing: {decision.get('endpoint')} "
                f"(confidence: {decision.get('confidence', 0):.4f})"
            )

            return decision

    def process_document(
        self,
        image,
        ocr_text: str,
        words: List[str],
        boxes: List[List[int]],
        schemas: Dict[str, Dict[str, Any]],
        api_schemas: List[Dict[str, Any]],
        pmg_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process document end-to-end through all stages.

        Args:
            image: Document image
            ocr_text: OCR text
            words: OCR words
            boxes: Bounding boxes
            schemas: Document type schemas
            api_schemas: SAP API schemas
            pmg_context: PMG context for similar cases

        Returns:
            Complete processing result
        """
        with Timer("Complete Document Processing"):
            result = {
                "success": False,
                "errors": [],
            }

            try:
                # Stage 3-4: Classification
                doc_type, subtype, class_conf = self.classify(
                    image,
                    ocr_text,
                    words,
                    boxes,
                )

                result["doc_type"] = doc_type
                result["subtype"] = subtype
                result["classification_confidence"] = class_conf

                # Get schema for document type
                schema = schemas.get(doc_type)
                if not schema:
                    raise ValueError(f"No schema found for {doc_type}")

                # Stage 5: Extraction
                extracted_data, extraction_metadata = self.extract(
                    image,
                    ocr_text,
                    words,
                    boxes,
                    doc_type,
                    schema,
                )

                result["extracted_data"] = extracted_data
                result["extraction_metadata"] = extraction_metadata

                # Stage 6: Quality Check (simplified)
                quality_score = self._check_quality(extracted_data, schema)
                result["quality_score"] = quality_score

                if quality_score < 0.90:
                    logger.warning(
                        f"Low quality score: {quality_score:.4f}, "
                        "attempting self-correction"
                    )
                    # TODO: Implement self-correction

                # Stage 7: Validation (simplified)
                violations = self._validate_business_rules(
                    extracted_data,
                    doc_type,
                )
                result["violations"] = violations

                # Stage 8: Routing
                if not violations:
                    similar_cases = pmg_context.get("similar_routings", []) if pmg_context else []

                    routing_decision = self.route(
                        extracted_data,
                        doc_type,
                        api_schemas,
                        similar_cases,
                    )

                    result["routing"] = routing_decision
                else:
                    result["routing"] = {
                        "next_action": "exception_handling",
                        "reason": "Business rule violations",
                    }

                result["success"] = True

            except Exception as e:
                logger.error(f"Document processing failed: {e}", exc_info=True)
                result["errors"].append(str(e))

            return result

    def _map_class_to_type(self, class_idx: int) -> str:
        """Map classification index to document type."""
        # TODO: Load from config
        doc_types = [
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

        if 0 <= class_idx < len(doc_types):
            return doc_types[class_idx]
        return "OTHER"

    def _identify_subtype(self, doc_type: str, ocr_text: str) -> str:
        """Identify document subtype (simplified)."""
        # TODO: Use dedicated subtype classifier
        # For now, return "STANDARD"
        return "STANDARD"

    def _check_quality(self, data: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """Check extraction quality (simplified)."""
        # TODO: Implement comprehensive quality checking
        required_fields = schema.get("required", [])

        if not required_fields:
            return 1.0

        present = sum(1 for field in required_fields if field in data and data[field])
        completeness = present / len(required_fields)

        return completeness

    def _validate_business_rules(
        self,
        data: Dict[str, Any],
        doc_type: str,
    ) -> List[Dict[str, Any]]:
        """Validate against business rules (simplified)."""
        # TODO: Implement comprehensive business rule validation
        violations = []

        # Example: Check required fields
        if doc_type == "SUPPLIER_INVOICE":
            if "total_amount" not in data or data["total_amount"] <= 0:
                violations.append({
                    "rule": "REQUIRED_FIELD",
                    "field": "total_amount",
                    "message": "Total amount is required and must be positive",
                })

        return violations

    def save(self, output_dir: str) -> None:
        """Save all model components."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving UnifiedExtractorModel to {output_dir}")

        if self.vision_encoder is not None:
            self.vision_encoder.save(str(output_path / "vision_encoder"))

        if self.language_decoder is not None:
            self.language_decoder.save(str(output_path / "language_decoder"))

        if self.reasoning_engine is not None:
            self.reasoning_engine.save(str(output_path / "reasoning_engine"))

        # Save config
        if self.config is not None:
            config_path = output_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(self.config.model_dump() if hasattr(self.config, 'model_dump') else {}, f, indent=2)

        logger.info("Model saved successfully")

    @classmethod
    def load(
        cls,
        model_dir: str,
        device: str = "cuda",
    ) -> "UnifiedExtractorModel":
        """Load all model components from directory."""
        model_path = Path(model_dir)

        logger.info(f"Loading UnifiedExtractorModel from {model_dir}")

        # Create instance
        model = cls(device=device)

        # Load components
        if (model_path / "vision_encoder").exists():
            model.vision_encoder = VisionEncoder.load(
                str(model_path / "vision_encoder"),
                device=device,
            )

        if (model_path / "language_decoder").exists():
            model.language_decoder = LanguageDecoder.load(
                str(model_path / "language_decoder"),
                device=device,
            )

        if (model_path / "reasoning_engine").exists():
            model.reasoning_engine = ReasoningEngine.load(
                str(model_path / "reasoning_engine"),
                device=device,
            )

        logger.info("Model loaded successfully")

        return model
