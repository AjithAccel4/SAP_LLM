"""
Enhanced Unified SAP_LLM Model - Production Ready.

Combines Vision Encoder, Language Decoder, and Reasoning Engine into a
single end-to-end model for document processing with:
- Comprehensive quality checking
- Dedicated subtype classification
- Self-correction mechanism
- Business rule validation
- Document type configuration
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from sap_llm.models.business_rule_validator import BusinessRuleValidator
from sap_llm.models.language_decoder import LanguageDecoder
from sap_llm.models.quality_checker import QualityChecker
from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.models.self_corrector import SelfCorrector
from sap_llm.models.subtype_classifier import SubtypeClassifier
from sap_llm.models.vision_encoder import VisionEncoder
from sap_llm.utils.logger import get_logger
from sap_llm.utils.timer import Timer

logger = get_logger(__name__)


class UnifiedExtractorModel(nn.Module):
    """
    Enhanced Unified SAP_LLM model combining all components.

    This model orchestrates the three main components plus quality assurance:
    1. Vision Encoder: Visual-text feature extraction
    2. Language Decoder: Structured JSON generation
    3. Reasoning Engine: Decision making and routing
    4. Quality Checker: Multi-dimensional quality assessment
    5. Subtype Classifier: Rule-based subtype detection
    6. Self-Corrector: Automatic error correction
    7. Business Rule Validator: Comprehensive validation

    Total Parameters: ~13.8B
    - Vision Encoder: 300M
    - Language Decoder: 7B
    - Reasoning Engine: 6B active (47B total)

    Args:
        config: Configuration object
        device: Device to run models on
        doc_types_config_path: Path to document types configuration
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        device: str = "cuda",
        doc_types_config_path: Optional[str] = None,
    ):
        super().__init__()

        self.config = config
        self.device = device

        logger.info("Initializing Enhanced UnifiedExtractorModel...")

        # Initialize ML components
        self.vision_encoder: Optional[VisionEncoder] = None
        self.language_decoder: Optional[LanguageDecoder] = None
        self.reasoning_engine: Optional[ReasoningEngine] = None

        # Initialize quality assurance components
        self.quality_checker = QualityChecker(confidence_threshold=0.70)
        self.subtype_classifier = SubtypeClassifier()
        self.self_corrector = SelfCorrector(
            confidence_threshold=0.70,
            max_attempts=2,
        )
        self.business_rule_validator = BusinessRuleValidator()

        # Load document types configuration
        self.doc_types_config = self._load_document_types_config(
            doc_types_config_path
        )

        # Load models if config provided
        if config is not None:
            self._load_models_from_config(config)

        logger.info("Enhanced UnifiedExtractorModel initialized")

    def _load_document_types_config(
        self,
        config_path: Optional[str],
    ) -> Dict[int, Dict[str, Any]]:
        """Load document types configuration from YAML."""
        if config_path is None:
            # Try default location
            default_path = Path(__file__).parent.parent.parent / "configs" / "document_types.yaml"
            if default_path.exists():
                config_path = str(default_path)
            else:
                logger.warning("Document types config not found, using fallback")
                return self._get_fallback_doc_types()

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded document types configuration from {config_path}")
                return config.get("document_types", {})
        except Exception as e:
            logger.error(f"Failed to load document types config: {e}")
            return self._get_fallback_doc_types()

    def _get_fallback_doc_types(self) -> Dict[int, Dict[str, Any]]:
        """Get fallback document types if config loading fails."""
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

        return {
            i: {"name": name, "subtypes": ["STANDARD"]}
            for i, name in enumerate(doc_types)
        }

    def _load_models_from_config(self, config: Any) -> None:
        """Load all ML models from configuration."""
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

            # Stage 4: Type Identifier - ENHANCED with dedicated classifier
            subtype, subtype_conf = self.subtype_classifier.classify(
                doc_type,
                ocr_text,
            )

            logger.info(
                f"Classification: {doc_type} / {subtype} "
                f"(confidence: {confidence:.4f}, subtype conf: {subtype_conf:.4f})"
            )

            return doc_type, subtype, min(confidence, subtype_conf)

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
        field_confidences: Optional[Dict[str, float]] = None,
        enable_self_correction: bool = True,
    ) -> Dict[str, Any]:
        """
        Process document end-to-end through all stages with enhanced quality assurance.

        Args:
            image: Document image
            ocr_text: OCR text
            words: OCR words
            boxes: Bounding boxes
            schemas: Document type schemas
            api_schemas: SAP API schemas
            pmg_context: PMG context for similar cases
            field_confidences: Optional field-level confidence scores
            enable_self_correction: Enable automatic self-correction

        Returns:
            Complete processing result with quality metrics
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

                # Stage 6: Quality Check - ENHANCED with comprehensive checker
                quality_assessment = self.quality_checker.check_quality(
                    extracted_data,
                    schema,
                    field_confidences,
                )
                result["quality_assessment"] = quality_assessment
                result["quality_score"] = quality_assessment["overall_score"]

                # ENHANCED: Self-Correction if quality is low
                if quality_assessment["overall_score"] < 0.90 and enable_self_correction:
                    logger.info(
                        f"Quality score {quality_assessment['overall_score']:.2f} below threshold, "
                        "attempting self-correction"
                    )

                    corrected_data, correction_metadata = self.self_corrector.correct(
                        extracted_data,
                        quality_assessment,
                        ocr_text,
                        schema,
                        pmg_context,
                    )

                    result["extracted_data"] = corrected_data
                    result["self_correction"] = correction_metadata

                    # Re-check quality after correction
                    quality_assessment = self.quality_checker.check_quality(
                        corrected_data,
                        schema,
                        field_confidences,
                    )
                    result["quality_assessment_post_correction"] = quality_assessment
                    result["quality_score"] = quality_assessment["overall_score"]

                    logger.info(
                        f"Post-correction quality: {quality_assessment['overall_score']:.2f}"
                    )

                # Stage 7: Validation - ENHANCED with comprehensive business rules
                violations = self.business_rule_validator.validate(
                    result["extracted_data"],
                    doc_type,
                    context=pmg_context,
                )
                result["violations"] = violations
                result["has_errors"] = any(v["severity"] == "ERROR" for v in violations)
                result["has_warnings"] = any(v["severity"] == "WARNING" for v in violations)

                # Stage 8: Routing
                if not result["has_errors"]:
                    similar_cases = pmg_context.get("similar_routings", []) if pmg_context else []

                    routing_decision = self.route(
                        result["extracted_data"],
                        doc_type,
                        api_schemas,
                        similar_cases,
                    )

                    result["routing"] = routing_decision
                else:
                    result["routing"] = {
                        "next_action": "exception_handling",
                        "reason": "Business rule violations (errors)",
                        "violations": [v for v in violations if v["severity"] == "ERROR"],
                    }

                result["success"] = True

            except Exception as e:
                logger.error(f"Document processing failed: {e}", exc_info=True)
                result["errors"].append(str(e))

            return result

    def _map_class_to_type(self, class_idx: int) -> str:
        """Map classification index to document type - ENHANCED with config."""
        # Load from configuration
        if class_idx in self.doc_types_config:
            return self.doc_types_config[class_idx]["name"]

        logger.warning(f"Unknown class index {class_idx}, returning OTHER")
        return "OTHER"

    def save(self, output_dir: str) -> None:
        """Save all model components."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving Enhanced UnifiedExtractorModel to {output_dir}")

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

        logger.info(f"Loading Enhanced UnifiedExtractorModel from {model_dir}")

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
